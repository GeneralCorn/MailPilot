from django.urls import path
from . import views

urlpatterns = [
    path("", views.inbox, name="inbox"),
    path("email/<int:idx>/", views.email_detail, name="email_detail"),
    path("email/<int:idx>/triage/", views.triage_email, name="triage_email"),
    path("import/", views.import_emails, name="import_emails"),
    path("delete-emails/", views.delete_emails, name="delete_emails"),
    path("traces/", views.traces, name="traces"),
]
