"""Emerging-issue detection must flag the week-2 app-update incident and stay quiet on week 1."""
import pandas as pd
import pytest

from orvana.config import settings
from orvana.report.emerging import emerging_issues


def week(name):
    df = pd.read_csv(settings.incoming_dir / name, dtype=str).dropna(subset=["message_text"]).drop_duplicates("ticket_id")
    return df.assign(category="app_issue")


@pytest.mark.parametrize("name,expected", [("tickets_2026-07-01_to_2026-07-07.csv", 0),
                                           ("tickets_2026-07-08_to_2026-07-14.csv", 1)])
def test_emerging(name, expected):
    issues = emerging_issues(week(name))
    assert len(issues) == expected
    if expected:
        assert {"update", "5.2"} <= set(issues[0]["terms"]) and issues[0]["tickets"] >= 55


def test_normal_vocabulary_is_not_new():
    df = pd.DataFrame({"ticket_id": [str(i) for i in range(10)], "created_at": ["2026-07-01"] * 10,
                       "category": ["delivery_delay"] * 10, "message_text": ["فين الطلب؟ قالولي هيوصل امبارح"] * 10})
    assert emerging_issues(df) == []
