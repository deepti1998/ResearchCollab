from django.shortcuts import render, redirect
# Create your views here.
from django.views.generic import TemplateView, ListView, CreateView
from django.db.models import Q
from .models import SearchBar, Book
from django.urls import reverse_lazy
from .forms import BookForm


class HomePageView(TemplateView):
    template_name = 'index.html'


class SearchResultsView(ListView):
    model = SearchBar
    template_name = 'search_results.html'

    def get_queryset(self):
        query = self.request.GET.get('q')
        print(query)
        searchable = SearchBar()
        searchable.set_query(query)
        search_term = SearchBar.objects.filter(
            Q(query__icontains=query)
        )
        return search_term


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
