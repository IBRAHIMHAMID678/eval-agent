"""The judge agent: grades a (question, answer, sources) triple.

How it works
------------
1. The answer's factual claims are extracted as atomic statements, each
   labeled SUPPORTED / CONTRADICTED / UNSUPPORTED with a source quote.
2. Three metrics are scored 1-5:
     - faithfulness: every claim grounded in the sources?
     - relevance:    does the answer address the question / follow instructions?
     - hallucination: scored inversely (5 = nothing invented, 1 = fabricated).
       The judge must quote the exact unsupported spans.
3. The model returns strict JSON (Groq JSON mode), validated by pydantic.
4. Token usage and wall-clock latency are captured on every call.

Also offers pairwise comparison: `compare()` pits two candidate answers
against each other and picks the better-grounded one.

All LLM traffic goes through Groq. The key is read from the GROQ_API_KEY
environment variable at call time -- never hardcoded, never stored.
"""

import json
import os
import time
from typing import List, Optional

from groq import Groq

from eval_agent.schemas import (
    ClaimVerdict,
    ComparisonVerdict,
    HallucinationResult,
    MetricScore,
    UsageStats,
    Verdict,
)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluation judge. Your job is to grade an AI assistant's \
answer against the source documents provided. You must NOT use any outside knowledge -- only what is written in \
the sources counts as ground truth.

Follow this procedure exactly:
1. Extract the answer's factual claims as a list of short atomic statements (one fact each). Skip pure \
opinions, greetings, and meta-text like "Here is the answer:".
2. For EACH claim, decide its status and give evidence:
   - "supported": a source states it, or it follows by direct logical entailment. Quote the supporting source text.
   - "contradicted": a source says the opposite. Quote the refuting source text.
   - "unsupported": no source covers it. Evidence is "".
3. Score the three metrics:
   - faithfulness (1-5): 5 = every claim supported; 4 = minor imprecision but nothing false; \
3 = some unsupported detail; 2 = key claims unsupported/contradicted; 1 = answer largely fabricated.
   - relevance (1-5): does the answer address the asked question and follow its explicit instructions \
(format, language, length)? 5 = fully on-point; 1 = answers a different question or ignores instructions.
   - hallucination (1-5, inverse scale): 5 = zero invented facts; 4 = trivial embellishment; \
3 = some invented specifics; 2 = major fabricated facts; 1 = answer is mostly invented.
4. Derive verdicts: score >= 4 -> "pass", score == 3 -> "borderline", score <= 2 -> "fail".

Respond with ONLY a JSON object in exactly this shape (no markdown fences, no commentary):
{
  "claims": [{"text": "atomic claim 1", "status": "supported", "evidence": "source quote..."},
             {"text": "atomic claim 2", "status": "unsupported", "evidence": ""}],
  "faithfulness": {"score": 5, "rationale": "..."},
  "relevance": {"score": 5, "rationale": "..."},
  "hallucination": {"score": 5, "rationale": "...", "unsupported_spans": []},
  "summary": "two-sentence plain-English summary of the judgement"
}
Be strict: an answer that sounds fluent but invents a date, name, or number the sources never mention \
must score faithfulness <= 2 and hallucination <= 2, with the invented span quoted in unsupported_spans."""

COMPARE_SYSTEM_PROMPT = """You are a strict, impartial evaluation judge. You are given a question, the source \
documents (your ONLY ground truth -- no outside knowledge), and TWO candidate answers labeled A and B.

Compare them on:
1. Faithfulness first: which answer's factual claims are better supported by the sources? An answer that \
invents dates, names, or numbers loses to a plainer answer that stays grounded.
2. Then relevance and completeness: which answers the question more directly and completely?

Respond with ONLY a JSON object in exactly this shape (no markdown fences, no commentary):
{
  "winner": "A",
  "score_a": 4,
  "score_b": 2,
  "rationale": "why the winner won, citing specific claims and sources",
  "deciding_factor": "the single most decisive difference in one sentence"
}
"winner" is "A", "B", or "tie". Scores are 1-5 overall quality for each answer."""


def _build_user_prompt(question: str, answer: str, sources: List[str]) -> str:
    docs = "\n\n".join(f"[Source {i + 1}]\n{doc}" for i, doc in enumerate(sources))
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{docs}\n\n"
        f"ANSWER TO GRADE:\n{answer}\n\n"
        "Grade this answer now. Return ONLY the JSON object."
    )


def _build_compare_prompt(
    question: str, answer_a: str, answer_b: str, sources: List[str]
) -> str:
    docs = "\n\n".join(f"[Source {i + 1}]\n{doc}" for i, doc in enumerate(sources))
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{docs}\n\n"
        f"ANSWER A:\n{answer_a}\n\n"
        f"ANSWER B:\n{answer_b}\n\n"
        "Compare the two answers now. Return ONLY the JSON object."
    )


def _label(score: int) -> str:
    if score >= 4:
        return "pass"
    if score == 3:
        return "borderline"
    return "fail"


class JudgeAgent:
    """Grades answers via a Groq-hosted LLM acting as judge."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it in your shell "
                "(export GROQ_API_KEY=...) -- it is never hardcoded or stored."
            )
        self.client = Groq(api_key=key)
        self.model = model
        self.temperature = temperature

    def _call(self, system: str, user: str) -> tuple[dict, UsageStats]:
        started = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(completion, "usage", None)
        stats = UsageStats(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
        raw = completion.choices[0].message.content or "{}"
        return json.loads(raw), stats

    def judge(
        self,
        question: str,
        answer: str,
        sources: List[str],
        case_id: Optional[str] = None,
    ) -> Verdict:
        """Score one answer. Returns a validated Verdict."""
        if not sources:
            raise ValueError("The judge needs at least one source document.")
        data, usage = self._call(
            JUDGE_SYSTEM_PROMPT, _build_user_prompt(question, answer, sources)
        )

        def metric(d: object) -> MetricScore:
            # Models occasionally return a bare score ("4") or a malformed
            # block; coerce instead of crashing the whole case.
            if not isinstance(d, dict):
                d = {"score": d}
            raw_score = d.get("score", 3)
            try:
                score = int(str(raw_score).strip().split("/")[0].split()[0])
            except (ValueError, TypeError, IndexError):
                score = 3
            score = max(1, min(5, score))
            return MetricScore(
                score=score,
                verdict=_label(score),  # type: ignore[arg-type]
                rationale=str(d.get("rationale", "")),
            )

        hall_d = data.get("hallucination") or {}
        if not isinstance(hall_d, dict):
            hall_d = {"score": hall_d}
        spans = hall_d.get("unsupported_spans", [])
        if isinstance(spans, str):
            spans = [spans]
        hallucination = HallucinationResult(
            **metric(hall_d).model_dump(),
            unsupported_spans=[str(s) for s in spans],
        )
        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            raw_claims = []
        claims = []
        for c in raw_claims:
            if isinstance(c, str):
                c = {"text": c, "status": "unsupported", "evidence": ""}
            if not isinstance(c, dict):
                continue
            claims.append(
                ClaimVerdict(
                    text=str(c.get("text", "")),
                    status=c.get("status", "unsupported"),  # type: ignore[arg-type]
                    evidence=str(c.get("evidence", "")),
                )
            )
        return Verdict.from_metrics(
            faithfulness=metric(data.get("faithfulness") or {}),
            relevance=metric(data.get("relevance") or {}),
            hallucination=hallucination,
            claims=claims,
            summary=str(data.get("summary", "")),
            model=self.model,
            usage=usage,
            case_id=case_id,
        )

    def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        sources: List[str],
    ) -> ComparisonVerdict:
        """Pairwise judgement: which of two answers is better grounded?"""
        if not sources:
            raise ValueError("The judge needs at least one source document.")
        data, usage = self._call(
            COMPARE_SYSTEM_PROMPT,
            _build_compare_prompt(question, answer_a, answer_b, sources),
        )
        return ComparisonVerdict(
            winner=data.get("winner", "tie"),  # type: ignore[arg-type]
            score_a=int(data.get("score_a", 3)),
            score_b=int(data.get("score_b", 3)),
            rationale=str(data.get("rationale", "")),
            deciding_factor=str(data.get("deciding_factor", "")),
            usage=usage,
            model=self.model,
        )
