from django.urls import path
from .views import HomePageView,  BookListView, UploadBookView,delete_book,book_list, get_queryset

urlpatterns = [
    path('books/', book_list, name='book_list'),
    path('books/<int:pk>/', delete_book, name='delete_book'),
    path('books/', BookListView.as_view(), name='class_book_list'),
    path('books/upload/', UploadBookView.as_view(), name='class_upload_book'),
    path('search_results/', get_queryset, name='search_results'),
    path('', HomePageView.as_view(), name='home')
]