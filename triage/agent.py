"""
MailPilot Agent Pipeline
========================
Router  → classify emails into categories
Evaluator → validate classification, gate risky emails
Ranker  → score and order the priority queue
Worker  → select and execute tool actions
"""
from .schemas import AgentMessage, Message

def build_router_messages(email: Message) -> list[AgentMessage]:
    """Build minimal prompt messages for the router stage."""
    return [
        AgentMessage(
            role="system",
            content=(
                "You are the Router in MailPilot. "
                "Classify the email into one of: marketing, personal, work, risk, billing, unclassified."
            ),
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

def route(email):
    """Classify an email into one of: Marketing, Personal, Work, Risk, Billing.
    Uses schema-constrained JSON prompting with few-shot examples.
    Returns dict with category, confidence, explanation."""
    # TODO: LLM call with structured output
    messages = build_router_messages(email)
    pass


def evaluate(email, classification):
    """Validate the router's classification.
    Re-classifies low-confidence emails. Flags risk items.
    Returns dict with final_category, risk_score, needs_review."""
    # TODO: confidence check + safety gate
    pass


def rank(emails):
    """Score and reorder the email batch by priority.
    Considers due dates, urgency cues, sender role, thread continuation.
    Returns sorted list of (email_id, priority_score)."""
    # TODO: feature-based priority scoring
    pass


def work(email, approved_actions):
    """Execute approved actions for an email using tool calls.
    Supports: label, flag, archive, reply_draft, calendar, escalate.
    Returns list of action results."""
    # TODO: tool-first prompting + bounded retries
    pass


def run_pipeline(email_batch):
    """Full triage pipeline: route → evaluate → rank → work.
    Processes a batch of emails end-to-end."""
    # TODO: orchestrate the four stages
    pass
