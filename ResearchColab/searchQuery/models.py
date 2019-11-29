from django.db import models

# Create your models here.
from django.db import models
from .validators import validate_file_extension


class SearchBar(models.Model):
    query = models.TextField(blank=False, null=False)

    def __str__(self):
        return self.query

    def set_query(self, query):
        self.query = query
        self.save()


class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    pdf = models.FileField(upload_to='books/pdfs/', validators=[validate_file_extension])
    cover = models.ImageField(upload_to='books/covers/', null=True, blank=True)

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.pdf.delete()
        self.cover.delete()
        super().delete(*args, **kwargs)
