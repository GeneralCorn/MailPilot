"""Gmail API integration — OAuth + fetch emails."""
import base64
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent.parent
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def get_gmail_service():
    """Authenticate and return a Gmail API service object."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=9090)
        TOKEN_FILE.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def _extract_part(payload, mime_type):
    """Recursively find a part by mime type."""
    if payload.get("mimeType") == mime_type and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        result = _extract_part(part, mime_type)
        if result:
            return result
    return ""


def fetch_emails(max_results=20):
    """Fetch recent emails from Gmail."""
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    emails = []
    for msg_info in messages:
        msg = service.users().messages().get(userId="me", id=msg_info["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        payload = msg["payload"]

        # Prefer HTML, fall back to plain text
        body_html = _extract_part(payload, "text/html")
        body_plain = _extract_part(payload, "text/plain")

        sender_name, sender_email = parseaddr(headers.get("From", ""))
        received_at = None
        date_str = headers.get("Date", "")
        if date_str:
            try:
                received_at = parsedate_to_datetime(date_str)
            except Exception:
                pass

        emails.append({
            "subject": headers.get("Subject", "(no subject)"),
            "sender": sender_email,
            "sender_name": sender_name,
            "recipient": parseaddr(headers.get("To", ""))[1],
            "body_html": body_html,
            "body_plain": body_plain,
            "snippet": msg.get("snippet", "")[:300],
            "received_at": received_at,
            "thread_id": msg.get("threadId", ""),
            "source": "gmail",
        })

    return emails
