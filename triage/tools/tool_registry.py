from storage import _load, _save


def label_email(email_id: str, label: str) -> dict:
    emails = _load()
    emails_by_id = {e.get("id"): e for e in emails}
    if email_id not in emails_by_id:
        raise KeyError(f"Email with id={email_id} not found")
    email = emails_by_id[email_id]
    email["category"] = label  
    _save(emails)
    return email

    