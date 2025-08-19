# verification/forms.py
from django import forms

class DocumentUploadForm(forms.Form):
    document = forms.ImageField(required=True)

class SelfieUploadForm(forms.Form):
    selfie = forms.ImageField(required=True)
