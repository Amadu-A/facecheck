# verification/urls.py
from django.urls import path
from .views import HomeView, UploadDocumentView, UploadSelfieView, AnalyzeView, CaptureSelfieView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("upload-document/", UploadDocumentView.as_view(), name="upload_document"),
    path("upload-selfie/", UploadSelfieView.as_view(), name="upload_selfie"),
    path("capture-selfie/", CaptureSelfieView.as_view(), name="capture_selfie"),
    path("analyze/", AnalyzeView.as_view(), name="analyze"),
]
