"""Test-set loader.

Each case in data/test_cases.json has:
  id, category, question, sources (list of docs), answer,
  expected: {faithfulness, relevance, hallucination, overall} verdict labels,
  notes: what the judge is expected to catch.

Run:  python -m eval_agent.testset --list
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PATH = DATA_DIR / "test_cases.json"


def load_cases(path: str = str(DEFAULT_PATH)) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    required = {"id", "category", "question", "sources", "answer", "expected"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case {case.get('id', '?')} missing fields: {missing}")
    return cases


def categories(cases: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for c in cases:
        if c["category"] not in seen:
            seen.append(c["category"])
    return seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the eval test set")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--list", action="store_true", help="list case ids by category")
    args = parser.parse_args()
    cases = load_cases(args.path)
    print(f"{len(cases)} cases, categories: {', '.join(categories(cases))}")
    if args.list:
        for cat in categories(cases):
            print(f"\n[{cat}]")
            for c in cases:
                if c["category"] == cat:
                    exp = c["expected"].get("overall", "?")
                    print(f"  {c['id']}  expected={exp}  :: {c['notes'][:90]}")
