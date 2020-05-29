from django.db import models

# Create your models here.
from django.db import models
import json
from .validators import validate_file_extension


class SearchBar(models.Model):
    query = models.TextField(blank=False, null=False)

    def __str__(self):
        return self.query

    def set_query(self, query):
        self.query = query
        self.save()


# uploaded into database
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


# authors papers
class DocumentDetails(models.Model):
    id = models.CharField(primary_key=True, max_length=500)
    abstract = models.TextField(blank=False, null=True)
    authors = models.TextField(blank=False, null=True)
    title = models.TextField(blank=False, null=True)
    references = models.TextField(blank=False, null=True)
    venue = models.TextField(blank=False, null=True)
    n_citation = models.IntegerField()
    year = models.IntegerField()

    def getAllPapersfromAuthors(authors):
        papers = []
        document_set = DocumentDetails.objects.all()
        for i in document_set:
            if len(set(json.loads(i.authors)).intersection(set(authors))) > 0:
                papers.append(i)
        return papers



