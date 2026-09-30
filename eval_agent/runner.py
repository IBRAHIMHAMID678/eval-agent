"""Batch runner: score the whole test set, aggregate, save JSON + Markdown report.

Usage:
    python -m eval_agent.runner                      # run all cases
    python -m eval_agent.runner --category adversarial
    python -m eval_agent.runner --out reports --model llama-3.3-70b-versatile

Outputs (under --out, default ./reports):
    results.json   one record per case: verdict + expected + match flags
    REPORT.md      summary table: pass rates per metric + per category,
                   plus per-case detail with mismatches highlighted
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval_agent.judge import JudgeAgent
from eval_agent.schemas import Verdict
from eval_agent.testset import categories, load_cases

METRICS = ["faithfulness", "relevance", "hallucination"]


def run_batch(cases, judge: JudgeAgent, delay: float = 1.0, verbose: bool = True):
    results = []
    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i}/{len(cases)}] judging {case['id']} ...", flush=True)
        try:
            verdict = judge.judge(
                question=case["question"],
                answer=case["answer"],
                sources=case["sources"],
                case_id=case["id"],
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            verdict, error = None, f"{type(exc).__name__}: {exc}"
        results.append({"case": case, "verdict": verdict, "error": error})
        if delay and i < len(cases):
            time.sleep(delay)
    return results


def _verdict_of(verdict: Verdict | None, metric: str) -> str:
    if verdict is None:
        return "error"
    if metric == "overall":
        return verdict.overall_verdict
    return getattr(verdict, metric).verdict


def aggregate(results) -> dict:
    """Pass rates per metric and per category, plus judge-vs-expected agreement."""
    agg: dict = {
        "total": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "by_metric": {},
        "by_category": {},
        "agreement": {},
    }
    for metric in METRICS + ["overall"]:
        counts = Counter(_verdict_of(r["verdict"], metric) for r in results)
        judged = sum(v for k, v in counts.items() if k != "error")
        agg["by_metric"][metric] = {
            "pass": counts["pass"],
            "borderline": counts["borderline"],
            "fail": counts["fail"],
            "pass_rate": round(counts["pass"] / judged, 3) if judged else 0.0,
        }
    cats = defaultdict(list)
    for r in results:
        cats[r["case"]["category"]].append(r)
    for cat, rs in cats.items():
        counts = Counter(_verdict_of(r["verdict"], "overall") for r in rs)
        judged = sum(v for k, v in counts.items() if k != "error")
        agg["by_category"][cat] = {
            "total": len(rs),
            "pass": counts["pass"],
            "fail": counts["fail"],
            "pass_rate": round(counts["pass"] / judged, 3) if judged else 0.0,
        }
    # agreement: how often did the judge match the expected verdict?
    for metric in METRICS + ["overall"]:
        matched = sum(
            1
            for r in results
            if r["verdict"] and r["case"]["expected"].get(metric) == _verdict_of(r["verdict"], metric)
        )
        judged = sum(1 for r in results if r["verdict"])
        agg["agreement"][metric] = {
            "matched": matched,
            "judged": judged,
            "rate": round(matched / judged, 3) if judged else 0.0,
        }
    return agg


def _md_verdict_cell(label: str) -> str:
    icon = {"pass": "✅", "borderline": "⚠️", "fail": "❌", "error": "💥"}.get(label, label)
    return f"{icon} {label}"


def write_report(results, agg: dict, out_dir: Path, model: str) -> Path:
    lines = [
        "# Eval Report",
        "",
        f"- **Model (judge):** `{model}`",
        f"- **Run at:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Cases:** {agg['total']} ({agg['errors']} errors)",
        "",
        "## Pass rates per metric",
        "",
        "| Metric | Pass | Borderline | Fail | Pass rate |",
        "|---|---|---|---|---|",
    ]
    for metric in METRICS + ["overall"]:
        m = agg["by_metric"][metric]
        lines.append(
            f"| {metric} | {m['pass']} | {m['borderline']} | {m['fail']} | {m['pass_rate']:.1%} |"
        )
    lines += [
        "",
        "## Pass rates per category",
        "",
        "| Category | Cases | Pass | Fail | Pass rate |",
        "|---|---|---|---|---|",
    ]
    for cat, c in agg["by_category"].items():
        lines.append(f"| {cat} | {c['total']} | {c['pass']} | {c['fail']} | {c['pass_rate']:.1%} |")
    lines += [
        "",
        "## Judge-vs-expected agreement",
        "",
        "How often the judge's verdict matched the verdict the test case was designed for.",
        "",
        "| Metric | Matched | Judged | Agreement |",
        "|---|---|---|---|",
    ]
    for metric in METRICS + ["overall"]:
        a = agg["agreement"][metric]
        lines.append(f"| {metric} | {a['matched']} | {a['judged']} | {a['rate']:.1%} |")
    lines += ["", "## Per-case detail", ""]
    for r in results:
        case, verdict, error = r["case"], r["verdict"], r["error"]
        exp = case["expected"]
        lines.append(f"### {case['id']} — {case['category']}")
        lines.append(f"> {case['notes']}")
        lines.append("")
        if error:
            lines.append(f"💥 **Error:** {error}")
        else:
            assert verdict is not None
            row = " | ".join(
                f"{m}: {_md_verdict_cell(_verdict_of(verdict, m))} (expected {exp.get(m, '?')})"
                for m in METRICS + ["overall"]
            )
            lines.append(row)
            lines.append("")
            for m in METRICS:
                ms = getattr(verdict, m)
                lines.append(f"- **{m}** ({ms.score}/5): {ms.rationale}")
            if verdict.hallucination.unsupported_spans:
                lines.append("- **Unsupported spans:**")
                for span in verdict.hallucination.unsupported_spans:
                    lines.append(f'  - "{span}"')
            lines.append(f"- **Summary:** {verdict.summary}")
        lines.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run the judge over the test set")
    parser.add_argument("--path", default=None, help="path to test_cases.json")
    parser.add_argument("--category", default=None, help="run only this category")
    parser.add_argument("--out", default="reports", help="output directory")
    parser.add_argument("--model", default=None, help="Groq model id for the judge")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between calls")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.path) if args.path else load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            raise SystemExit(f"No cases in category '{args.category}'. Known: {', '.join(categories(load_cases(args.path) if args.path else load_cases()))}")
    print(f"Running {len(cases)} cases...")

    judge_kwargs = {}
    if args.model:
        judge_kwargs["model"] = args.model
    judge = JudgeAgent(**judge_kwargs)

    results = run_batch(cases, judge, delay=args.delay, verbose=not args.quiet)
    agg = aggregate(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "case_id": r["case"]["id"],
            "category": r["case"]["category"],
            "expected": r["case"]["expected"],
            "verdict": r["verdict"].model_dump() if r["verdict"] else None,
            "error": r["error"],
        }
        for r in results
    ]
    (out_dir / "results.json").write_text(
        json.dumps({"model": judge.model, "aggregate": agg, "results": serializable}, indent=2),
        encoding="utf-8",
    )
    report_path = write_report(results, agg, out_dir, judge.model)

    print("\nPass rates:")
    for metric in METRICS + ["overall"]:
        m = agg["by_metric"][metric]
        print(f"  {metric:13s} {m['pass_rate']:.1%} pass ({m['pass']}/{agg['total'] - agg['errors']})")
    print(f"\nWrote {(out_dir / 'results.json')} and {report_path}")


if __name__ == "__main__":
    main()
