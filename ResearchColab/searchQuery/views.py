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
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from skcriteria import Data,MIN,MAX
from skcriteria.madm import closeness, simple
spacy.load('en_core_web_sm')
parser = English()
en_stop = set(nltk.corpus.stopwords.words('english'))
aminer_dir = 'G:/Major project/aminer-academic-citation-dataset'


with open(os.path.join(aminer_dir, 'AMiner-Author.txt'), 'r',encoding="utf8") as f:
    dict_list = []
    c_dict = {}
    for i, line in enumerate(f):
        c_line = line.strip()[1:].strip()
        if len(c_line)<1:
            if len(c_dict)>0:
                dict_list += [c_dict]
            c_dict = {}
        else:
            c_frag = c_line.split(' ')
            c_dict[c_frag[0]] = ' '.join(c_frag[1:])

criteria = [MAX, MAX, MAX, MAX]
author_df1 = pd.DataFrame(dict_list)
author_df1.rename({'index': 'Authorid',
                 'a': 'Affiliation',
                 'n': 'Author',
                 'pc': 'NPapers',
                 'cn': 'Citations',
                  'hi': 'Hindex',
                  't': 'Research Interests'
                 }, axis=1, inplace=True)
author_df1.drop(author_df1.columns[[2,7,9, 10]], axis = 1, inplace = True)
author_df = author_df1.head(720000)
ldamodel = gensim.models.ldamulticore.LdaMulticore.load('G:/Major project/lda_modelupdated_new.model')
mm = corpora.Dictionary.load('G:/Major project/imdb.dict')
en_stop = set(nltk.corpus.stopwords.words('english'))
b = np.loadtxt('G:/Major project/npstack.txt', dtype=np.float64)


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


def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix
    #     print(p)
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))


# In[35]:

def get_most_similar_documents(query, matrix, k=15):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query, matrix)  # list of jensen shannon distances
    #     print(sims[:15])
    return sims.argsort()[:k]

class HomePageView(TemplateView):
    template_name = 'index.html'


def get_queryset(request):
        new_doc = request.GET.get('q')
        print(new_doc)
        new_doc = prepare_text_for_lda(new_doc)
        new_bow = mm.doc2bow(new_doc)
        new_doc_distribution = np.array([tup[1] for tup in ldamodel.get_document_topics(bow=new_bow)])
        most_sim_ids = get_most_similar_documents(new_doc_distribution, b)
        most_similar_df = author_df[author_df.index.isin(most_sim_ids)]
        most_similar_df1 = most_similar_df[["Author", "NPapers","Hindex","pi"]]
        print(most_similar_df1)
        features_df = most_similar_df.filter(['Authorid', 'NPapers', 'Citations', 'Hindex', 'pi'], axis=1)
        features_df.NPapers = features_df.NPapers.astype(float)
        features_df.Citations = features_df.Citations.astype(float)
        features_df.Hindex = features_df.Hindex.astype(float)
        features_df.pi = features_df.pi.astype(float)
        features_df["NPapers"] = features_df["NPapers"] / features_df["NPapers"].max()
        features_df["Citations"] = features_df["Citations"] / features_df["Citations"].max()
        features_df["Hindex"] = features_df["Hindex"] / features_df["Hindex"].max()
        features_df["pi"] = features_df["pi"] / features_df["pi"].max()
        features_df.set_index('Authorid')
        ft = features_df.copy()
        # features_df = features_df.head(100)
        # ft = ft.head(100)
        ft.drop(ft.columns[[0]], axis=1, inplace=True)
        mtx = ft.values.tolist()
        data = Data(mtx, criteria,
                    weights=[.15, .09, .05, .07],
                    anames=features_df['Authorid'].values.tolist(),
                    cnames=["NPapers", "Citations", "Hindex", "pi"])
        dm = closeness.TOPSIS()
        dec = dm.decide(data)
        rank = dec.rank_
        most_similar_df1.insert(4, "Rank", rank, True)
        authors_similar = most_similar_df1.sort_values(by='Rank')
        authors_similar.drop(['Rank'], axis=1)
        authors_similar = authors_similar.reset_index()
        authors_similar.reindex(index=list(range(1, 16)))
        authors_similar.drop('index',axis=1,inplace=True)

        html_table = authors_similar.to_html(classes="table table-striped table-hover")
        context = {
            'html_table': html_table
        }
        return render(request,'search_results.html',context)


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






