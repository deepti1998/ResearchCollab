from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import SearchBar


class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ("query",)


admin.site.register(SearchBar, SearchQueryAdmin)

