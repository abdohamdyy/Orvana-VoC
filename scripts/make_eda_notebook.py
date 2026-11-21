"""Generate notebooks/01_eda.ipynb (Stage 1 evidence). Execute with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# Stage 1: Exploratory data analysis\nEvery number quoted in `docs/STAGE1_DATA.md` is reproduced here."),
    code("import sys, re, json\nsys.path.insert(0, '..'); sys.path.insert(0, '../scripts')\nimport pandas as pd\n"
         "from orvana.normalize import match_key, clean_message\nfrom orvana.extractors import split_segments, find_product, find_store\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_colwidth', 90)\n"
         "t = pd.read_csv('../data/tickets.csv'); g = pd.read_csv('../data/tickets_dev_gold.csv')\n"
         "w1 = pd.read_csv('../incoming/tickets_2026-07-01_to_2026-07-07.csv'); w2 = pd.read_csv('../incoming/tickets_2026-07-08_to_2026-07-14.csv')\n"
         "print(t.shape, g.shape, w1.shape, w2.shape)"),
    md("## 1. Basic profile"),
    code("print(t.isna().sum().to_dict())\nprint('labelled:', t.label_category.notna().sum())\nprint(t.label_category.value_counts().to_dict())\n"
         "ar = t.message_text.str.contains(r'[\\u0621-\\u064A]')\n"
         "franco = ~ar & t.message_text.str.contains(r'[a-z][2357]|[2357][a-z]|\\b(?:el|w|men|fel|law|ya|mesh|talabt|gali)\\b', case=False)\n"
         "print('Arabic %.3f  Franco %.3f  English %.3f' % (ar.mean(), franco.mean(), (~ar & ~franco).mean()))\n"
         "print(t.channel.value_counts().to_dict())"),
    md("## 2. Leakage: the gold set is inside tickets.csv"),
    code("print('gold ids in tickets.csv:', g.ticket_id.isin(t.ticket_id).sum())\n"
         "keys = set(g.message_text.map(match_key))\nother = t[~t.ticket_id.isin(g.ticket_id)]\n"
         "print('other rows sharing a (normalised) gold text:', other.message_text.map(match_key).isin(keys).sum())"),
    md("## 3. Duplicates and templates"),
    code("print('exact duplicate texts:', t.message_text.duplicated().sum())\n"
         "print(t.message_text.str.extract(r'^([^,،]{1,20}[,،])')[0].value_counts().head(8).to_dict())\n"
         "print('connector share:', round(t.message_text.str.contains(r'وكمان|w kaman|and also').mean(), 3))\n"
         "print('boilerplate tail share:', round(t.message_text.str.contains('أنا عميل عندكم|ana 3amil|loyal customer').mean(), 3))"),
    md("## 4. Staff-label noise (template-core majority vote)\n`scripts/build_train_set.py` groups messages by core and compares each staff label with the core's majority label."),
    code("print(json.dumps(json.load(open('../outputs/metrics/label_noise.json')), indent=1, ensure_ascii=False))\n"
         "seg = pd.read_csv('../data/processed/segments.csv')\n"
         "print(seg.groupby('core').size().sort_values(ascending=False).head(12))"),
    md("## 5. Secondary category = text after the connector"),
    code("two = g.message_text.map(lambda x: len(split_segments(x)) == 2)\n"
         "print(pd.crosstab(two, g.secondary_category.notna(), rownames=['connector split'], colnames=['gold has secondary']))"),
    md("## 6. Product and store: dictionary matching on the gold"),
    code("n = lambda s: s.fillna('none').astype(str)\n"
         "print('product', (n(g.message_text.map(find_product)) == n(g.product_mentioned)).mean())\n"
         "print('store  ', (n(g.message_text.map(find_store)) == n(g.store_mentioned)).mean())"),
    md("## 7. Urgency rules (see docs/URGENCY_RULES.md)"),
    code("from orvana.rules import urgency, sentiment\nfrom orvana.schema import Category\n"
         "pred = [urgency(Category(r.category), split_segments(clean_message(r.message_text).text)[0], clean_message(r.message_text).text) for r in g.itertuples()]\n"
         "print('urgency rules vs gold (true category):', (pd.Series(pred) == g.urgency).mean())\n"
         "print(g.groupby('urgency').category.value_counts().unstack(fill_value=0))\n"
         "print('sentiment:', g.sentiment.value_counts().to_dict(), '| neutral categories:', g[g.sentiment=='neutral'].category.unique())"),
    md("## 8. Prompt injection in the data"),
    code("allm = pd.concat([t.message_text, w1.message_text, w2.message_text.dropna()])\n"
         "inj = allm[allm.map(lambda x: clean_message(x).injection_flag)]\nprint(len(inj)); print(inj.str[-90:].tolist())"),
    md("## 9. Incoming weeks"),
    code("for name, w in [('week1', w1), ('week2', w2)]:\n"
         "    print(name, len(w), 'dup ids', w.ticket_id.duplicated().sum(), 'empty', w.message_text.isna().sum(),\n"
         "          'sorted', pd.to_datetime(w.created_at).is_monotonic_increasing, 'no customer_id', w.customer_id.isna().sum())\n"
         "t['created_at'] = pd.to_datetime(t.created_at)\nprint('history tickets/week (mean):', round(t.set_index('created_at').resample('W').size().mean()))"),
    md("## 10. Something new in week 2"),
    code("from orvana.report.emerging import emerging_issues\nfor name, w in [('week1', w1), ('week2', w2)]:\n"
         "    v = w.dropna(subset=['message_text']).drop_duplicates('ticket_id').assign(category='?')\n"
         "    print(name, [(e['terms'][:4], e['tickets']) for e in emerging_issues(v)])"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks" / "01_eda.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote notebooks/01_eda.ipynb")
