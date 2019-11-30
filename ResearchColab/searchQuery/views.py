import json
import os
import numpy as np
import pandas as pd
from itertools import cycle, chain
from scipy.stats import entropy
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import pickle
from gensim import corpora
import spacy
from spacy.lang.en import English
from django.shortcuts import render, redirect
# Create your views here.
from django.views.generic import TemplateView, ListView, CreateView
from django.db.models import Q
from .models import SearchBar, Book, DocumentDetails
from django.urls import reverse_lazy
from .forms import BookForm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn


spacy.load('en_core_web_sm')
parser = English()
en_stop = set(nltk.corpus.stopwords.words('english'))


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


class HomePageView(TemplateView):
    template_name = 'index.html'


class SearchResultsView(ListView):
    model = SearchBar
    template_name = 'search_results.html'

    def get_queryset(self):
        new_doc = self.request.GET.get('q')
        print(new_doc)
        new_doc = prepare_text_for_lda(new_doc)
        new_bow = mm.doc2bow(new_doc)

        return new_doc


class BookListView(ListView):
    model = Book
    template_name = 'class_book_list.html'
    context_object_name = 'books'


class UploadBookView(CreateView):
    model = Book
    form_class = BookForm
    success_url = reverse_lazy('class_book_list')
    template_name = 'upload_book.html'


def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {
        'books': books
    })


def delete_book(request, pk):
    if request.method == 'POST':
        book = Book.objects.get(pk=pk)
        book.delete()
    return redirect('book_list')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# from .models import DocumentDetails
def extractdocumentdetails():
    file = 'D:/ResearchColab/ResearchColab/static/papers/dblp-ref-0.json'
    with open(file) as f:
        for line in f:
            result = json.loads(line)
            abstract = result.get("abstract")
            id = result.get("id")
            authors = json.dumps(result["authors"])
            n_citation = result.get("n_citation")
            references = json.dumps(result.get("references"))
            venue = result.get("venue")
            year = result.get("year")
            title = result.get("title")
            DocumentDetails.objects.create(abstract=abstract, id=id, authors=authors, n_citation=n_citation,
                                           references=references, venue=venue, year=year, title=title)






