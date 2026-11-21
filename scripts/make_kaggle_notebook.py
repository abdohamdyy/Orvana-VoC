"""Generate notebooks/03_train_kaggle.ipynb from scripts/train_transformer.py (keeps them in sync)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
script = (ROOT / "scripts" / "train_transformer.py").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


cells = [
    md("# Stage 3: fine-tune a multilingual encoder (Kaggle / Colab, free T4)\n\n"
       "1. Settings: **Accelerator = GPU T4**, **Internet = on**.\n"
       "2. Upload `data/processed/segments.csv` (made by `scripts/build_train_set.py`): on Kaggle as a dataset, "
       "on Colab with the file panel.\n"
       "3. Run all. Download `stage3_model.zip` and unzip it into `models/stage3/` in the repo.\n\n"
       "The gold set is never used: `build_train_set.py` removed every gold ticket before making `segments.csv`."),
    code("!pip -q install onnx onnxruntime\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code("import glob\nDATA = (glob.glob('/kaggle/input/**/segments.csv', recursive=True) + glob.glob('/content/segments.csv') "
         "+ glob.glob('segments.csv'))[0]\nprint(DATA)"),
    code("%%writefile train_transformer.py\n" + script),
    md("## Main model: xlm-roberta-base"),
    code("!python train_transformer.py --data $DATA --model xlm-roberta-base --out stage3_model"),
    md("## Smaller alternative for the comparison: multilingual-e5-small (faster on CPU)"),
    code("!python train_transformer.py --data $DATA --model intfloat/multilingual-e5-small --out stage3_e5small"),
    code("import json\nfor d in ['stage3_model', 'stage3_e5small']:\n"
         "    m = json.load(open(f'{d}/metrics.json'))\n"
         "    print(d, 'best_epoch', m['best_epoch'], 'dev macro-F1', m['selection_history'][m['best_epoch']-1]['dev_macro_f1'],"
         " 'minutes', m['train_minutes_total'])"),
    code("!cd stage3_model && zip -q -r ../stage3_model.zip model.int8.onnx tokenizer.json labels.json metrics.json train_log.txt\n"
         "!cd stage3_e5small && zip -q -r ../stage3_e5small.zip model.int8.onnx tokenizer.json labels.json metrics.json train_log.txt\n"
         "!ls -la *.zip"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "03_train_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
