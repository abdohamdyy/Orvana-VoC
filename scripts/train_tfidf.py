"""Train the TF-IDF baseline. Reports internal-dev accuracy (train split only), then refits on
train+dev for the saved model.   python scripts/train_tfidf.py"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orvana.backends.tfidf import TfidfBackend  # noqa: E402
from orvana.config import settings  # noqa: E402

seg = pd.read_csv(settings.data_dir / "processed" / "segments.csv")
seg = seg[seg.label_clean.notna() & seg.text.notna()]
tr, dev = seg[seg.split == "train"], seg[seg.split == "dev"]

model = TfidfBackend.train(tr.text.tolist(), tr.label_clean.tolist())
pred = [c.value for c, _ in model.classify(dev.text.tolist())]
print(classification_report(dev.label_clean, pred, digits=3))
metrics = {"internal_dev_macro_f1": f1_score(dev.label_clean, pred, average="macro"), "n_dev_segments": len(dev),
           "n_train_segments": len(tr)}

final = TfidfBackend.train(seg.text.tolist(), seg.label_clean.tolist())
final.save()
(settings.outputs_dir / "metrics" / "tfidf_train.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
