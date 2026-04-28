import json
import os

import anthropic

from triage.schemas import AgentMessage, Category, Message, State

_MODEL = "claude-sonnet-4-6"
_LOW_CONFIDENCE_THRESHOLD = 0.65
_HIGH_RISK_THRESHOLD = 0.6

_SYSTEM = (
    "You are the Evaluator in MailPilot. Review the Router's email classification.\n"
    "Verify the category, assess risk, and output your own confidence score.\n\n"
    "Categories: marketing, personal, work, risk, billing, unclassified\n\n"
    "Set needs_review=true if: category is 'risk', risk_score > 0.6, or classification is ambiguous.\n\n"
    "Respond with valid JSON only, no other text:\n"
    '{"final_category": "<category>", "confidence": <0.0-1.0>, '
    '"risk_score": <0.0-1.0>, "needs_review": <true|false>, "explanation": "<brief reason>"}'
)


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
            content='{"final_category": "marketing", "confidence": 0.96, "risk_score": 0.02, "needs_review": false, "explanation": "Confirmed: standard newsletter, no risk indicators"}',
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
            content='{"final_category": "personal", "confidence": 0.91, "risk_score": 0.01, "needs_review": false, "explanation": "Overriding router: casual tone and personal Gmail address indicate personal email"}',
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
            content='{"final_category": "risk", "confidence": 0.93, "risk_score": 0.92, "needs_review": true, "explanation": "Overriding router: suspicious sender domain and urgency tactics indicate phishing despite billing framing"}',
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
            content='{"final_category": "personal", "confidence": 0.92, "risk_score": 0.05, "needs_review": false, "explanation": "Overriding router: legitimate welcome email from verified google.com domain, account setup language is not a phishing indicator here"}',
        ),
        AgentMessage(role="user", content=email_content),
    ]


def evaluate(email: Message, state: State) -> None:
    """Verify classification and write risk_score into state; append to needs_review if flagged."""
    category = state.classifications.get(email.id, Category.UNCLASSIFIED)
    confidence = state.confidence_scores.get(email.id, 0.0)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    all_msgs = _build_messages(email, category, confidence)
    system = next(m.content for m in all_msgs if m.role == "system")
    messages = [{"role": m.role, "content": m.content} for m in all_msgs if m.role != "system"]

    raw = ""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=512,
                system=system,
                messages=messages,
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            final_category = Category(data["final_category"])
            new_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            risk_score = max(0.0, min(1.0, float(data.get("risk_score", 0.0))))
            needs_review = bool(data.get("needs_review", False))

            if risk_score > _HIGH_RISK_THRESHOLD or new_confidence < _LOW_CONFIDENCE_THRESHOLD:
                needs_review = True

            state.classifications[email.id] = final_category
            state.confidence_scores[email.id] = new_confidence
            state.risk_scores[email.id] = risk_score
            if needs_review and email.id not in state.needs_review:
                state.needs_review.append(email.id)
            return
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON or unknown category. Respond only with valid JSON:\n"
                            '{"final_category": "<marketing|personal|work|risk|billing|unclassified>", '
                            '"confidence": <0.0-1.0>, "risk_score": <0.0-1.0>, '
                            '"needs_review": <true|false>, "explanation": "<brief reason>"}'
                        ),
                    },
                ]
        except anthropic.APIError:
            break

    state.risk_scores[email.id] = 0.0
    if email.id not in state.needs_review:
        state.needs_review.append(email.id)
