import json
import os

import anthropic

from triage.schemas import AgentMessage, Category, Message

_MODEL = "claude-haiku-4-5"

_SYSTEM = (
    "You are the Router in MailPilot. Classify the email into exactly one category:\n"
    "- marketing: Promotional emails, newsletters, advertisements, discount offers\n"
    "- personal: Emails from friends, family, or personal acquaintances\n"
    "- work: Professional emails, meetings, projects, business correspondence\n"
    "- risk: Phishing, scams, suspicious links, security threats, high-risk content\n"
    "- billing: Invoices, receipts, payment confirmations, subscription charges\n"
    "- unclassified: Does not clearly fit any of the above categories\n\n"
    "Respond with valid JSON only, no other text:\n"
    '{"category": "<category>", "confidence": <0.0-1.0>, "explanation": "<brief reason>"}'
)


def build_messages(email: Message) -> list[AgentMessage]:
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
            content='{"category": "marketing", "confidence": 0.97, "explanation": "Promotional discount email from a retail sender"}',
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
            content='{"category": "work", "confidence": 0.95, "explanation": "Internal meeting invitation from a business colleague"}',
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
            content='{"category": "risk", "confidence": 0.94, "explanation": "Phishing attempt with urgency tactics and suspicious sender domain"}',
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


def route(email: Message) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    all_msgs = build_messages(email)
    system = next(m.content for m in all_msgs if m.role == "system")
    messages = [{"role": m.role, "content": m.content} for m in all_msgs if m.role != "system"]

    raw = ""
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=256,
                system=system,
                messages=messages,
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return {
                "category": Category(data["category"]),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                "explanation": str(data.get("explanation", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON or unknown category. Respond only with valid JSON:\n"
                            '{"category": "<marketing|personal|work|risk|billing|unclassified>", '
                            '"confidence": <0.0-1.0>, "explanation": "<brief reason>"}'
                        ),
                    },
                ]
        except anthropic.APIError:
            break

    return {"category": Category.UNCLASSIFIED, "confidence": 0.0, "explanation": "classification_failed"}
