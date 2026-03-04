from storage import _load, _save


def label_email(email_id: int, label: str) -> dict:
    emails = _load()
    emails_by_id = {e['id']: e for e in emails}
    email = emails_by_id[emails_by_id]
    email['label']


    

    