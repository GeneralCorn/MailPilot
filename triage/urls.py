from django.urls import path
from . import views

urlpatterns = [
    path("", views.inbox, name="inbox"),
    path("email/<int:pk>/", views.email_detail, name="email_detail"),
    path("email/<int:pk>/triage/", views.triage_email, name="triage_email"),
    path("task/<int:pk>/approve/", views.approve_task, name="approve_task"),
    path("task/<int:pk>/execute/", views.execute_task, name="execute_task"),
    path("import/", views.import_emails, name="import_emails"),
    path("run-pipeline/", views.run_pipeline, name="run_pipeline"),
]
