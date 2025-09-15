# app/utils/scoring.py
from typing import List, Dict

DEFAULT_DEDUCTIONS = {
    "LOOK_AWAY": 2,
    "NO_FACE": 5,
    "MULTIPLE_FACES": 10,
    "HAND_NEAR_FACE": 12,
    "PHONE_SUSPECTED": 15,
    "EYE_CLOSED": 5,
}

def compute_integrity_score(events: List[Dict]) -> Dict:
    counts = {}
    total_deduction = 0
    for e in events:
        et = e.get("event_type")
        counts[et] = counts.get(et, 0) + 1
        deduction = DEFAULT_DEDUCTIONS.get(et, 1)
        total_deduction += deduction
    score = max(0, 100 - total_deduction)
    return {"score": score, "counts": counts, "total_deductions": total_deduction}
