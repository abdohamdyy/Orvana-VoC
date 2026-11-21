"""Stage 3 backend: the fine-tuned encoder exported to ONNX int8. CPU only, no torch needed.

Expects in settings.model_dir (default models/stage3/): model.int8.onnx, tokenizer.json, labels.json
(the zip produced by notebooks/03_train_kaggle.ipynb).
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from ..config import settings
from ..schema import Category


class FinetunedBackend:
    def __init__(self, model_dir: Path):
        meta = json.loads((model_dir / "metrics.json").read_text()) if (model_dir / "metrics.json").exists() else {}
        self.labels = json.loads((model_dir / "labels.json").read_text())
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=64)
        pad = "<pad>" if self.tok.token_to_id("<pad>") is not None else "[PAD]"  # XLM-R vs BERT vocab
        self.tok.enable_padding(pad_id=self.tok.token_to_id(pad), pad_token=pad)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model_dir / "model.int8.onnx"), opts, providers=["CPUExecutionProvider"])
        self.name = f"finetuned:{meta.get('base_model', model_dir.name)}-int8"
        calib = model_dir / "calibration.json"  # temperature fitted on the internal dev set
        self.temperature = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "FinetunedBackend":
        return cls(model_dir or settings.model_dir)

    def probabilities(self, segments: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(segments), batch_size):
            # same preprocessing as scripts/train_transformer.py::prep (case carries no category signal)
            enc = self.tok.encode_batch([s.lower() for s in segments[i : i + batch_size]])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            logits = self.session.run(["logits"], feeds)[0] / self.temperature
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(exp / exp.sum(axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def logits(self, segments: list[str]) -> np.ndarray:
        """Raw (uncalibrated) logits, used to fit the temperature."""
        t, self.temperature = self.temperature, 1.0
        try:
            return np.log(self.probabilities(segments) + 1e-12)
        finally:
            self.temperature = t

    def classify(self, segments: list[str]) -> list[tuple[Category, float]]:
        probs = self.probabilities(segments)
        return [(Category(self.labels[i]), float(p[i])) for p, i in zip(probs, probs.argmax(axis=1))]
