from django.urls import path
from . import views

urlpatterns = [
    path("", views.inbox, name="inbox"),
    path("email/<int:idx>/", views.email_detail, name="email_detail"),
    path("email/<int:idx>/triage/", views.triage_email, name="triage_email"),
    path("email/<int:idx>/test-calendar/", views.test_calendar, name="test_calendar"),
    path("email/<int:idx>/test-send/", views.test_send, name="test_send"),
    path("import/", views.import_emails, name="import_emails"),
    path("run-pipeline/", views.run_pipeline, name="run_pipeline"),
]
