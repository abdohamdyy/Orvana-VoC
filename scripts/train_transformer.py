"""Stage 3: fine-tune a multilingual encoder to classify one complaint segment into 9 categories.

Runs on a free Kaggle/Colab T4 in minutes.  Usage:
    python train_transformer.py --data segments.csv --model xlm-roberta-base --out stage3_model

1. Train on the 'train' cores, evaluate every epoch on the held-out 'dev' cores (unseen templates),
   keep the best epoch count.
2. Re-train on train+dev with that epoch count (the final model sees every template).
3. Export to ONNX + int8 quantisation, so inference needs only onnxruntime on a CPU.
Never sees tickets_dev_gold.csv (removed upstream by build_train_set.py).
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["delivery_delay", "wrong_item", "refund_request", "product_quality", "price_complaint",
          "app_issue", "staff_behavior", "loyalty_points", "inquiry"]
L2I = {l: i for i, l in enumerate(LABELS)}


def seed_all(seed: int):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)


def typo(text: str, rng: random.Random) -> str:
    """Mimic the typos in the data (dropped / doubled / glued characters) for robustness."""
    chars = list(text)
    for _ in range(rng.randint(1, 2)):
        if len(chars) < 4:
            break
        op = rng.choice(["drop", "double", "glue"])
        if op == "glue":
            spaces = [i for i, c in enumerate(chars) if c == " "]
            if spaces:
                chars.pop(rng.choice(spaces))
            continue
        i = rng.randrange(1, len(chars) - 1)
        if op == "drop":
            chars.pop(i)
        else:  # double
            chars.insert(i, chars[i])
    return "".join(chars)


GREETINGS = ["السلام عليكم،", "مساء الخير،", "لو سمحت", "يا جماعة", "hi,", "hello,", "salam,", "law sama7t"]
SIGN_OFFS = ["شكرا", "ياريت حد يكلمني", "ارجو الرد", "thanks", "please reply", "plz rod", "shokran", "!!!"]


def augment(text: str, rng: random.Random) -> str:
    """Label-preserving noise so the model cannot lean on surface cues of the historical templates
    (every inquiry there ends with '؟'; every message has the same greetings)."""
    if rng.random() < 0.3:
        text = typo(text, rng)
    if rng.random() < 0.15:
        text = re.sub(r"[^\w\s]", "", text)  # no punctuation (voice-note style)
    if rng.random() < 0.15:
        text = f"{rng.choice(GREETINGS)} {text} {rng.choice(SIGN_OFFS)}"
    return text


def prep(text: str) -> str:
    """Same preprocessing at training and inference time (orvana/backends/finetuned.py)."""
    return text.lower()


def make_loader(texts, labels, tok, bs, shuffle, augment_on=False, seed=0):
    rng = random.Random(seed)

    def collate(batch):
        t = [prep(augment(x, rng) if augment_on else x) for x, _ in batch]
        enc = tok(t, truncation=True, max_length=64, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([int(y) for _, y in batch], dtype=torch.long)
        return enc

    return DataLoader(list(zip(texts, labels)), batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, tok, texts, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([prep(t) for t in texts[i : i + bs]], truncation=True, max_length=64, padding=True,
                  return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def train(texts, labels, args, device, eval_set=None, epochs=None, log=print):
    seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), ignore_mismatched_sizes=True).to(device)  # new 9-class head
    y = np.array([L2I[l] for l in labels])
    weights = torch.tensor(len(y) / (len(LABELS) * np.bincount(y, minlength=len(LABELS))), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))  # class weights for macro-F1

    loader = make_loader(texts, y, tok, args.batch_size, shuffle=True, augment_on=True, seed=args.seed)
    epochs = epochs or args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * epochs * len(loader)), epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_t = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), labels_t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        row = {"epoch": ep, "train_loss": round(total / len(loader), 4), "seconds": round(time.time() - t0, 1)}
        for name, (ev_texts, ev_labels) in (eval_set or {}).items():
            pred = [LABELS[i] for i in predict(model, tok, ev_texts, device).argmax(1)]
            row[f"{name}_macro_f1"] = round(f1_score(ev_labels, pred, average="macro"), 4)
        if eval_set:  # selection score = mean macro-F1 over the evaluation sets
            row["select_score"] = round(float(np.mean([row[f"{n}_macro_f1"] for n in eval_set])), 4)
        history.append(row)
        log(json.dumps(row))
    return model, tok, history


def export_onnx(model, tok, out: Path, log=print):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = model.float().cpu().eval()
    dummy = tok(["مثال للتصدير example"], return_tensors="pt")
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "logits": {0: "b"}})
    try:
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), out / "model.onnx", dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), out / "model.onnx", **kwargs)
    quantize_dynamic(out / "model.onnx", out / "model.int8.onnx", weight_type=QuantType.QInt8)
    (out / "model.onnx").unlink()  # keep only the small int8 model
    log(f"exported {out / 'model.int8.onnx'} ({(out / 'model.int8.onnx').stat().st_size / 1e6:.0f} MB)")


def onnx_parity(model, tok, out: Path, texts: list[str], log=print) -> float:
    """Share of texts where the int8 ONNX model predicts the same class as the torch model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "model.int8.onnx"), providers=["CPUExecutionProvider"])
    torch_pred = predict(model.float().cpu(), tok, texts, "cpu").argmax(1)
    onnx_pred = []
    for i in range(0, len(texts), 64):
        enc = tok([prep(t) for t in texts[i : i + 64]], truncation=True, max_length=64, padding=True, return_tensors="np")
        onnx_pred.append(sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                               "attention_mask": enc["attention_mask"].astype(np.int64)})[0].argmax(1))
    agree = float((np.concatenate(onnx_pred) == torch_pred).mean())
    log(f"ONNX int8 vs torch agreement on {len(texts)} texts: {agree:.4f}")
    return agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="segments.csv")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--out", default="stage3_model")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--extra-data", default=None, help="synthetic.csv (text,label,kept): new phrasings")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train_log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} gpu={torch.cuda.get_device_name(0) if device == 'cuda' else '-'} args={vars(args)}")
    t_start = time.time()

    seg = pd.read_csv(args.data)
    seg = seg[seg.label_clean.notna() & seg.text.notna()].drop_duplicates("text")
    tr, dev = seg[seg.split == "train"], seg[seg.split == "dev"]
    log(f"train segments={len(tr)} dev segments (held-out templates)={len(dev)}")
    tr_texts, tr_labels = tr.text.tolist(), tr.label_clean.tolist()
    evals = {"dev": (dev.text.tolist(), dev.label_clean.tolist())}
    extra = None
    if args.extra_data:  # synthetic new phrasings: 85% train, 15% held out for epoch selection
        extra = pd.read_csv(args.extra_data)
        extra = extra[extra.kept.astype(bool)].drop_duplicates("text").sample(frac=1.0, random_state=args.seed)
        cut = int(0.85 * len(extra))
        ex_tr, ex_dev = extra.iloc[:cut], extra.iloc[cut:]
        tr_texts, tr_labels = tr_texts + ex_tr.text.tolist(), tr_labels + ex_tr.label.tolist()
        evals["synth_dev"] = (ex_dev.text.tolist(), ex_dev.label.tolist())
        log(f"extra synthetic: {len(ex_tr)} train + {len(ex_dev)} selection")

    # 1) model selection on held-out templates (+ held-out synthetic phrasings)
    model, tok, hist = train(tr_texts, tr_labels, args, device, eval_set=evals, log=log)
    best_ep = max(hist, key=lambda r: (r["select_score"], -r["epoch"]))["epoch"]
    probs = predict(model, tok, dev.text.tolist(), device)
    report = classification_report(dev.label_clean, [LABELS[i] for i in probs.argmax(1)], digits=3, output_dict=True)
    log(f"best epoch = {best_ep}")
    del model
    torch.cuda.empty_cache()

    # 2) final model on all labelled data (all historical segments + all synthetic)
    all_texts, all_labels = seg.text.tolist(), seg.label_clean.tolist()
    if extra is not None:
        all_texts, all_labels = all_texts + extra.text.tolist(), all_labels + extra.label.tolist()
    final, tok, final_hist = train(all_texts, all_labels, args, device, epochs=best_ep, log=log)
    final.save_pretrained(out / "hf")
    tok.save_pretrained(out / "hf")
    tok.save_pretrained(out)  # tokenizer.json next to the ONNX file

    # 3) ONNX int8 for CPU inference (+ check it still predicts like the torch model)
    export_onnx(final, tok, out, log=log)
    parity = onnx_parity(final, tok, out, dev.text.tolist()[:500], log=log)
    (out / "labels.json").write_text(json.dumps(LABELS))
    metrics = {"base_model": args.model, "extra_data": args.extra_data, "onnx_int8_parity": parity,
               "selection_history": hist, "best_epoch": best_ep,
               "dev_report": report, "final_history": final_hist,
               "train_minutes_total": round((time.time() - t_start) / 60, 1),
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"total time {metrics['train_minutes_total']} min")


if __name__ == "__main__":
    main()
