from django.urls import path

from .views import HomePageView, SearchResultsView, BookListView, UploadBookView,delete_book,book_list

urlpatterns = [
    path('books/', book_list, name='book_list'),
    path('books/<int:pk>/', delete_book, name='delete_book'),
    path('books/', BookListView.as_view(), name='class_book_list'),
    path('books/upload/', UploadBookView.as_view(), name='class_upload_book'),
    path('search_results/', SearchResultsView.as_view(), name='search_results'),
    path('', HomePageView.as_view(), name='home')
]