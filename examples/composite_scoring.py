"""Composite scoring: one judgment split into atomic Score questions, weighted in code (weights are yours to change).
   python examples/composite_scoring.py [model]"""
import sys
sys.path.insert(0, ".")
from decider.infer import Decider

d = Decider(sys.argv[1] if len(sys.argv) > 1 else "Mapika/decider-2b")
QUESTIONS = {
    "severity": {"type": "score", "instructions": "How severe is the reported issue?",
                 "criteria": ["Cosmetic; no impact to functionality", "Broken or degraded feature, but a workaround exists", "Blocking issue; no workaround exists"]},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]},
    "actionable": {"type": "score", "instructions": "How much does the report give an engineer to work with?",
                   "criteria": ["no details", "some details (what happened)", "clear reproduction steps or error messages"]},
}
WEIGHTS = {"severity": 0.6, "frustration": 0.25, "actionable": 0.15}
reports = ["The export button is misaligned by a few pixels on the settings page.",
           "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari. Console says TypeError: undefined is not an object (exportBtn.js:41).",
           "NOBODY on our team can log in since this morning!!! 500 error on every attempt. We are losing money, this is unacceptable."]
for r in reports:
    a = d.system_one(r, QUESTIONS)["answers"]
    prio = sum(WEIGHTS[k] * a[k]["score"] / (len(QUESTIONS[k]["criteria"]) - 1) for k in WEIGHTS)       # each score normalised to 0..1
    print(f"priority {prio:.2f}  " + "  ".join(f"{k}={a[k]['score']:.2f} (p={a[k]['confidence']:.2f})" for k in WEIGHTS) + f"  | {r[:60]}")
