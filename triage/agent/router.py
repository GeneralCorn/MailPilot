import os

import anthropic

from triage.schemas import AgentMessage, Category, Message, State

from ._anthropic import call_tools
from ._caching import to_cached_request

_MODEL = "claude-sonnet-4-6"

_SYSTEM = (
    "You are the Router in MailPilot. Classify the incoming email into exactly one category:\n"
    "- marketing: Promotional emails, newsletters, advertisements, discount offers\n"
    "- personal: Emails from friends, family, or personal acquaintances\n"
    "- work: Professional emails, meetings, projects, business correspondence\n"
    "- risk: Phishing, scams, suspicious links, security threats, high-risk content\n"
    "- billing: Invoices, receipts, payment confirmations, subscription charges\n"
    "- unclassified: Does not clearly fit any of the above categories\n\n"
    "Use the classify_email tool to record your classification with a confidence score."
)

_TOOL = {
    "name": "classify_email",
    "description": "Record the router's classification of an email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": "The router's chosen category",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Router's confidence in the classification, 0.0 to 1.0",
            },
            "explanation": {
                "type": "string",
                "description": "One-line reason for the classification",
            },
        },
        "required": ["category", "confidence", "explanation"],
    },
}


def build_messages(email: Message, feedback: str | None = None) -> list[AgentMessage]:
    return [
        AgentMessage(role="system", content=_SYSTEM),
        AgentMessage(
            role="user",
            content=(
                "Subject: Flash Sale — 50% off this weekend only!\n"
                "Sender: deals@shopnow.com\n"
                "Body: Don't miss our biggest sale of the year. Limited time offer, shop now!"
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                'I will classify this email. classify_email('
                'category="marketing", confidence=0.97, '
                'explanation="Promotional discount email from a retail sender")'
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: Q3 planning meeting — Thursday 3 pm\n"
                "Sender: sarah.chen@acme.com\n"
                "Body: Hi team, please join us for our quarterly planning session in Conference Room B."
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                'classify_email(category="work", confidence=0.95, '
                'explanation="Internal meeting invitation from a business colleague")'
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: URGENT: Your account will be suspended\n"
                "Sender: support@secure-banking-verify.net\n"
                "Body: Your account has been flagged. Click here immediately to verify or face suspension."
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                'classify_email(category="risk", confidence=0.94, '
                'explanation="Phishing attempt with urgency tactics and suspicious sender domain")'
            ),
        ),
        AgentMessage(
            role="user",
            content=(
                "Subject: Review your Google Account settings for your new account\n"
                "Sender: no-reply@google.com\n"
                "Body: Welcome to Google. Here are a few tips to get you started. Control your account: review and adjust your privacy and security settings any time."
            ),
        ),
        AgentMessage(
            role="assistant",
            content=(
                'classify_email(category="personal", confidence=0.93, '
                'explanation="Legitimate account welcome email from verified Google domain, not phishing")'
            ),
        ),
        *(
            [
                AgentMessage(
                    role="user",
                    content=(
                        f"Your previous classification was rejected by the Evaluator. "
                        f"Reason: {feedback}. Reclassify carefully."
                    ),
                )
            ]
            if feedback
            else []
        ),
        AgentMessage(
            role="user",
            content=(
                f"Subject: {email.subject}\n"
                f"Sender: {email.sender}\n"
                f"Body: {email.body_plain or email.snippet}"
            ),
        ),
    ]


def route(email: Message, state: State, *, feedback: str | None = None) -> None:
    """Classify email and write results into state.classifications and state.confidence_scores."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    all_msgs = build_messages(email, feedback=feedback)
    system, messages = to_cached_request(all_msgs)

    result = call_tools(
        client,
        model=_MODEL,
        system=system,
        messages=messages,
        tools=[_TOOL],
        force_tool="classify_email",
        max_tokens=512,
    )

    if not result:
        state.classifications[email.id] = Category.UNCLASSIFIED
        state.confidence_scores[email.id] = 0.0
        return

    data = result[0]["input"]
    try:
        state.classifications[email.id] = Category(data["category"])
        state.confidence_scores[email.id] = max(
            0.0, min(1.0, float(data.get("confidence", 0.5)))
        )
    except (KeyError, ValueError):
        state.classifications[email.id] = Category.UNCLASSIFIED
        state.confidence_scores[email.id] = 0.0
