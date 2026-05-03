from dataclasses import dataclass

import anthropic

from triage.schemas import AgentMessage, Category, Message, State

from ._anthropic import call_tools
from ._caching import to_cached_request
from ._client import client_kwargs, default_model
LOW_CONFIDENCE_THRESHOLD = 0.65
HIGH_RISK_THRESHOLD = 0.6


@dataclass(frozen=True)
class EvaluatorResult:
    final_category: Category
    confidence: float
    risk_score: float
    needs_review: bool
    explanation: str
    reasoning: str = ""


_SYSTEM = (
    "You are the Evaluator in MailPilot. Review the Router's classification of an email,\n"
    "verify it against the email content, assess risk, and submit your verdict via the\n"
    "verify_classification tool.\n\n"
    "Categories: marketing, personal, work, risk, billing, unclassified\n\n"
    "Reason step-by-step BEFORE filling in the tool input. Your `reasoning` parameter\n"
    "MUST come first and walk through:\n"
    "  1. What category does the email content actually look like?\n"
    "  2. Does that match the Router's category? If not, why is the override justified?\n"
    "  3. Are there phishing or risk indicators (suspicious sender domain, urgency tactics,\n"
    "     unverified links, mismatched branding)? Or are surface-level red flags actually\n"
    "     legitimate (e.g. account-setup emails from real google.com)?\n"
    "  4. Is the Router's confidence justified by the evidence?\n"
    "Once you have reasoned through these, fill in the remaining fields consistent with\n"
    "that reasoning.\n\n"
    "Set needs_review=true if: category is 'risk', risk_score > 0.6, or classification\n"
    "is ambiguous."
)


_TOOL = {
    "name": "verify_classification",
    "description": "Submit the evaluator's verdict on the router's classification.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step analysis covering the 4 points in the system prompt",
            },
            "final_category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": "Final category — may override the router's pick",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Evaluator's own confidence in the final_category",
            },
            "risk_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Risk score (phishing, fraud, social engineering)",
            },
            "needs_review": {
                "type": "boolean",
                "description": "True if a human should review before any action runs",
            },
            "explanation": {
                "type": "string",
                "description": "One-line summary of the verdict",
            },
        },
        "required": [
            "reasoning",
            "final_category",
            "confidence",
            "risk_score",
            "needs_review",
            "explanation",
        ],
    },
}


def _build_messages(email: Message, category: Category, confidence: float) -> list[AgentMessage]:
    router_summary = f"Router classified as '{category.value}' (confidence {confidence:.2f})"
    email_content = (
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Body: {email.body_plain or email.snippet}\n\n"
        f"Router output: {router_summary}"
    )
    return [
        AgentMessage(role="system", content=_SYSTEM),
        AgentMessage(
            role="user",
            content=(
                "Subject: Your weekly newsletter is here\n"
                "Sender: news@techblog.io\n"
                "Body: This week in tech: AI breakthroughs and more.\n\n"
                "Router output: Router classified as 'marketing' (confidence 0.95)"
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                "verify_classification(reasoning=\"Subject and body explicitly call themselves "
                "a newsletter. Sender is a tech-blog domain doing periodic digest content. No "
                "urgency, no payment requests, no credential pages. Router said marketing with "
                "high confidence and the evidence agrees.\", final_category=\"marketing\", "
                "confidence=0.96, risk_score=0.02, needs_review=false, "
                "explanation=\"Confirmed standard newsletter\")"
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: Dinner plans for Saturday?\n"
                "Sender: mike.johnson@gmail.com\n"
                "Body: Hey, are you free Saturday? Thinking of trying that new Italian place.\n\n"
                "Router output: Router classified as 'work' (confidence 0.52)"
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                "verify_classification(reasoning=\"Body is conversational ('Hey, are you free') "
                "with a personal logistics ask about dinner. Sender is a personal gmail.com "
                "address, not a corporate domain. Router said work with low confidence (0.52) — "
                "the low confidence is itself a signal it was unsure. Casual tone + personal "
                "sender clearly point to personal. No risk indicators.\", "
                "final_category=\"personal\", confidence=0.91, risk_score=0.01, "
                "needs_review=false, explanation=\"Override to personal: casual tone + personal gmail\")"
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: Invoice #8821 — payment required\n"
                "Sender: billing@amaz0n-secure-payments.ru\n"
                "Body: Your account owes $299. Click to pay now or service will be suspended.\n\n"
                "Router output: Router classified as 'billing' (confidence 0.71)"
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                "verify_classification(reasoning=\"Surface framing is billing (Invoice, payment "
                "required), which is why Router picked billing. But the sender domain is "
                "amaz0n-secure-payments.ru — a typosquat (amaz0n with a zero) on a .ru TLD, "
                "definitely not Amazon. Body uses urgency tactics ('Click to pay now or service "
                "will be suspended') which is a classic phishing pattern. The legitimate billing "
                "reading is wrong; this is phishing impersonating billing.\", "
                "final_category=\"risk\", confidence=0.93, risk_score=0.92, "
                "needs_review=true, explanation=\"Phishing impersonating Amazon billing\")"
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: Review your Google Account settings for your new account\n"
                "Sender: no-reply@google.com\n"
                "Body: Welcome to Google. Here are a few tips to get you started. Review and adjust your privacy and security settings any time.\n\n"
                "Router output: Router classified as 'risk' (confidence 0.91)"
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                "verify_classification(reasoning=\"Router flagged risk because account-setup "
                "language can resemble phishing. But the sender is no-reply@google.com (the "
                "verified primary domain, not a lookalike), the body is a generic onboarding "
                "welcome with no payment request and no urgency, and the account-settings link "
                "is to legitimate Google help pages. Surface-level red flags here are "
                "explainable. This is a real Google account setup notice, not phishing.\", "
                "final_category=\"personal\", confidence=0.92, risk_score=0.05, "
                "needs_review=false, explanation=\"Legitimate Google onboarding\")"
            ),
        ),
        AgentMessage(role="user", content=email_content),
    ]


def evaluate(email: Message, state: State) -> EvaluatorResult:
    """Verify the router's classification, write back to state, and return the structured result."""
    category = state.classifications.get(email.id, Category.UNCLASSIFIED)
    confidence = state.confidence_scores.get(email.id, 0.0)

    client = anthropic.Anthropic(**client_kwargs())
    all_msgs = _build_messages(email, category, confidence)
    system, messages = to_cached_request(all_msgs)

    result = call_tools(
        client,
        model=default_model(),
        system=system,
        messages=messages,
        tools=[_TOOL],
        force_tool="verify_classification",
        # CoT reasoning eats tokens
        max_tokens=1024,
    )

    if not result:
        state.risk_scores[email.id] = 0.0
        return EvaluatorResult(
            final_category=state.classifications.get(email.id, Category.UNCLASSIFIED),
            confidence=state.confidence_scores.get(email.id, 0.0),
            risk_score=0.0,
            needs_review=True,
            explanation="evaluator failed to produce valid output",
        )

    data = result[0]["input"]
    try:
        final_category = Category(data["final_category"])
        new_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        risk_score = max(0.0, min(1.0, float(data.get("risk_score", 0.0))))
        needs_review = bool(data.get("needs_review", False))
        explanation = str(data.get("explanation", ""))
        reasoning = str(data.get("reasoning", ""))
    except (KeyError, ValueError):
        state.risk_scores[email.id] = 0.0
        return EvaluatorResult(
            final_category=state.classifications.get(email.id, Category.UNCLASSIFIED),
            confidence=state.confidence_scores.get(email.id, 0.0),
            risk_score=0.0,
            needs_review=True,
            explanation="evaluator returned malformed tool input",
        )

    if risk_score > HIGH_RISK_THRESHOLD or new_confidence < LOW_CONFIDENCE_THRESHOLD:
        needs_review = True

    state.classifications[email.id] = final_category
    state.confidence_scores[email.id] = new_confidence
    state.risk_scores[email.id] = risk_score

    return EvaluatorResult(
        final_category=final_category,
        confidence=new_confidence,
        risk_score=risk_score,
        needs_review=needs_review,
        explanation=explanation,
        reasoning=reasoning,
    )
