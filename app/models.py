from django.db import models
from django.contrib.auth.models import User

class UploadedDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class PreprocessingResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_rows = models.IntegerField()
    rows_after_cleaning = models.IntegerField()
    processed_at = models.DateTimeField(auto_now_add=True)
 
