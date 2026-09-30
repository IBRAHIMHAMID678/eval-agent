"""Structured verdict models for the judge agent.

Every judgement comes back as a Verdict: three per-metric scores (1-5),
per-claim support verdicts, usage stats, plus an overall verdict.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

VerdictLabel = Literal["pass", "borderline", "fail"]
ClaimStatus = Literal["supported", "contradicted", "unsupported"]


class MetricScore(BaseModel):
    """One scored metric: 1-5 score, derived verdict label, short rationale."""

    score: int = Field(ge=1, le=5, description="Integer score, 1 (worst) to 5 (best)")
    verdict: VerdictLabel = Field(description="pass (>=4), borderline (3), fail (<=2)")
    rationale: str = Field(description="One or two sentences explaining the score")


class HallucinationResult(MetricScore):
    """Hallucination metric plus the exact unsupported spans the judge found."""

    unsupported_spans: List[str] = Field(
        default_factory=list,
        description="Exact quotes from the answer that no source supports",
    )


class ClaimVerdict(BaseModel):
    """One atomic claim from the answer and how it fared against the sources."""

    text: str = Field(description="The atomic claim, as extracted by the judge")
    status: ClaimStatus = Field(
        description="supported / contradicted / unsupported by the sources"
    )
    evidence: str = Field(
        default="",
        description="Short quote from a source backing or refuting the claim ('' if none)",
    )


class UsageStats(BaseModel):
    """What the judgement cost in model resources."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


class Verdict(BaseModel):
    """Full judgement for one (question, answer, sources) triple."""

    faithfulness: MetricScore = Field(
        description="Is every factual claim in the answer grounded in the sources?"
    )
    relevance: MetricScore = Field(
        description="Does the answer address the question (and follow its instructions)?"
    )
    hallucination: HallucinationResult = Field(
        description="Are there invented facts? Scored inversely: 5 = clean, 1 = full of fabrications"
    )
    claims: List[ClaimVerdict] = Field(
        default_factory=list,
        description="Atomic claims extracted from the answer, each with a support verdict",
    )
    overall_score: float = Field(description="Mean of the three metric scores")
    overall_verdict: VerdictLabel = Field(
        description="pass if all metrics pass, fail if any fails, else borderline"
    )
    summary: str = Field(description="Two-sentence plain-English summary of the judgement")
    usage: UsageStats = Field(default_factory=UsageStats)
    model: str = Field(description="Judge model id that produced this verdict")
    case_id: Optional[str] = Field(default=None, description="Test case id, if judged as part of a batch")
    demo: bool = Field(default=False, description="True if this is canned demo output (no live LLM call)")

    @classmethod
    def from_metrics(
        cls,
        faithfulness: MetricScore,
        relevance: MetricScore,
        hallucination: HallucinationResult,
        claims: List[ClaimVerdict],
        summary: str,
        model: str,
        usage: Optional[UsageStats] = None,
        case_id: Optional[str] = None,
        demo: bool = False,
    ) -> "Verdict":
        overall = round(
            (faithfulness.score + relevance.score + hallucination.score) / 3, 2
        )
        labels = [faithfulness.verdict, relevance.verdict, hallucination.verdict]
        if all(label == "pass" for label in labels):
            overall_verdict: VerdictLabel = "pass"
        elif any(label == "fail" for label in labels):
            overall_verdict = "fail"
        else:
            overall_verdict = "borderline"
        return cls(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            claims=claims,
            overall_score=overall,
            overall_verdict=overall_verdict,
            summary=summary,
            usage=usage or UsageStats(),
            model=model,
            case_id=case_id,
            demo=demo,
        )


class ComparisonVerdict(BaseModel):
    """Result of a pairwise A/B comparison between two candidate answers."""

    winner: Literal["A", "B", "tie"] = Field(description="Which answer is better grounded")
    score_a: int = Field(ge=1, le=5, description="Overall quality of answer A, 1-5")
    score_b: int = Field(ge=1, le=5, description="Overall quality of answer B, 1-5")
    rationale: str = Field(description="Why the winner won, grounded in the sources")
    deciding_factor: str = Field(description="The single most decisive difference")
    usage: UsageStats = Field(default_factory=UsageStats)
    model: str = Field(description="Judge model id that produced this verdict")
    demo: bool = Field(default=False, description="True if canned demo output")
