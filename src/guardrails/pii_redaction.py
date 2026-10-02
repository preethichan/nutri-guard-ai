"""
Input guardrail: redact identifying PII before it is sent to the LLM, kept in
conversation history, or written to any log file.

This directly targets the information-leakage failure mode measured in
docs/eval_reports/baseline_v2 -- specifically the finding that 100% of
conversations where a user disclosed PII stored it completely unredacted in
the backend conversation log (see docs/baseline_evidence/04_backend_storage_leak_log.png).

Design choice -- redact identity, keep clinical context:
    Entities that identify a specific person (name, email, phone, SSN, credit
    card, location, medical license) are redacted. Entities that are
    clinically relevant but NOT identifying on their own (medication name,
    condition name, lab value) are deliberately preserved, because:
      1. They materially improve answer quality (e.g. a statin-grapefruit
         interaction caveat requires knowing the medication).
      2. Re-identification risk from "takes atorvastatin, triglycerides 310"
         alone (with no name/contact info attached) is low in this context.
    This mirrors the PII vs PHI-without-identifier distinction already drawn
    in src/eval/metrics.py's PII_ENTITIES list, and the eval harness measures
    the two leak categories separately (see src/eval/run_eval.py) so this
    tradeoff is reported transparently, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.metrics import get_presidio_analyzer

# Entities that identify a specific individual -> redacted.
REDACT_ENTITIES: dict[str, str] = {
    "PERSON": "[NAME]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "US_SSN": "[SSN]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "LOCATION": "[LOCATION]",
    "MEDICAL_LICENSE": "[MEDICAL_LICENSE]",
}

# Clinically-relevant entities that are intentionally NOT redacted -- kept in
# sync with src/eval/metrics.py's custom recognizers, for reference/testing.
PRESERVED_CLINICAL_ENTITIES = {"MEDICATION", "MEDICAL_CONDITION", "LAB_VALUE"}

_MIN_SCORE_BY_ENTITY = {
    "PERSON": 0.8,
    "LOCATION": 0.8,
}
_DEFAULT_MIN_SCORE = 0.5


@dataclass
class RedactionResult:
    original_text: str
    redacted_text: str
    findings: list[dict] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return len(self.findings) > 0


def redact_pii(text: str) -> RedactionResult:
    """Redact identifying PII from `text`, preserving clinical context terms."""
    analyzer = get_presidio_analyzer()
    results = analyzer.analyze(
        text=text, language="en", entities=list(REDACT_ENTITIES)
    )

    # Replace right-to-left so earlier span offsets stay valid.
    spans = sorted(
        (
            r
            for r in results
            if r.entity_type in REDACT_ENTITIES
            and r.score >= _MIN_SCORE_BY_ENTITY.get(r.entity_type, _DEFAULT_MIN_SCORE)
        ),
        key=lambda r: r.start,
        reverse=True,
    )

    redacted = text
    findings = []
    for r in spans:
        fragment = redacted[r.start : r.end]
        placeholder = REDACT_ENTITIES[r.entity_type]
        redacted = redacted[: r.start] + placeholder + redacted[r.end :]
        findings.append({"entity_type": r.entity_type, "score": r.score, "text": fragment})

    findings.reverse()  # restore left-to-right order for readability
    return RedactionResult(original_text=text, redacted_text=redacted, findings=findings)
