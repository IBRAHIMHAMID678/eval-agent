"""eval-agent: an LLM-as-judge harness for grading chatbot / RAG answers."""

from eval_agent.schemas import (
    ClaimVerdict,
    ComparisonVerdict,
    HallucinationResult,
    MetricScore,
    UsageStats,
    Verdict,
)
from eval_agent.judge import JudgeAgent

__all__ = [
    "JudgeAgent",
    "Verdict",
    "ComparisonVerdict",
    "MetricScore",
    "HallucinationResult",
    "ClaimVerdict",
    "UsageStats",
]
__version__ = "0.1.0"
