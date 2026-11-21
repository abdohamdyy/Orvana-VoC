"""The suite's perturbations must be deterministic (reproducible numbers) and actually change the text."""
from orvana.eval.perturb import PERTURBATIONS, perturb_all
from orvana.normalize import clean_message

ROWS = [("A", "الأوردر متأخر بقاله 5 أيام ومحدش بيرد"), ("B", "My Kiriazi fridge stopped cooling after 2 days")]


def test_deterministic():
    assert perturb_all(ROWS) == perturb_all(ROWS)


def test_every_transform_changes_something():
    names = {name for _, name, _ in perturb_all(ROWS)}
    assert len(names) >= len(PERTURBATIONS) - 1  # upper_latin cannot change pure Arabic text


def test_invisible_tricks_are_undone_by_normalisation():
    for name in ("zero_width", "bidi_marks", "tashkeel", "tatweel"):
        variants = [t for _, n, t in perturb_all(ROWS) if n == name and _.startswith("A")]
        assert clean_message(variants[0]).text == clean_message(ROWS[0][1]).text, name
