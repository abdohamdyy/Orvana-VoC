"""Choose CHAT_ANSWER_THRESHOLD from data, not by hand.

In-scope: the first turn of every DEV policy question / no-data question in tests/chat/scenarios.jsonl.
Off-topic: generic probes below (written for this purpose, not part of the scenarios).
The threshold is the midpoint between the highest off-topic score and the lowest in-scope score;
if they overlap, it is the value that minimises the errors on both lists.

    python scripts/calibrate_chat_threshold.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orvana.chat.kb import load_kb  # noqa: E402

OFF_TOPIC = [
    "what's the weather like today?", "who won the match yesterday?", "tell me a joke", "can you help with my math homework?",
    "ايه رأيك في الحكومة؟", "الجو عامل ايه النهارده؟", "مين كسب الماتش امبارح؟", "احكيلي نكتة", "ممكن تكتبلي قصيدة؟",
    "enta esmak eh?", "el gaw 3amel eh enharda?", "meen kesb el match?", "what is the capital of France?",
    "translate this sentence to French", "how do I bake a cake?", "ازاي اعمل كيكة؟", "ممكن تعلمني انجليزي؟",
    "what is 2 + 2?", "are you a robot?", "انت بني ادم ولا روبوت؟",
]


def main():
    kb = load_kb()
    scen = [json.loads(line) for line in (Path(__file__).resolve().parent.parent / "tests" / "chat" / "scenarios.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    in_scope = [s["turns"][0] for s in scen if s["split"] == "dev" and s["type"] in ("policy_q", "unknown")]
    ins = sorted(kb.search(t)[0][1] for t in in_scope)
    off = sorted(kb.search(t)[0][1] for t in OFF_TOPIC)
    print(f"in-scope  n={len(ins)}  min={ins[0]:.3f}  p10={ins[len(ins) // 10]:.3f}")
    print(f"off-topic n={len(off)}  max={off[-1]:.3f}  p90={off[int(0.9 * (len(off) - 1))]:.3f}")
    if off[-1] < ins[0]:
        tau = (off[-1] + ins[0]) / 2
    else:  # overlap: minimise misclassified questions on both sides
        cands = sorted(set(ins + off))
        tau = min(cands, key=lambda t: sum(s < t for s in ins) + sum(s >= t for s in off))
    print(f"threshold = {tau:.3f}")
    for t in OFF_TOPIC:
        f, s = kb.search(t)[0]
        if s >= tau:
            print(f"  off-topic above threshold: {t!r} -> {f.id} {s:.3f}")


if __name__ == "__main__":
    main()
