from django.conf.urls import url
from . import views

app_name="collector"
urlpatterns = [
    # Examples:
    
    
    url(r'^$', views.index, name='index'),
   	url(r'^(?P<tweet_id>[0-9]+)/$', views.detail, name='detail'),
    url(r'^(?P<tweet_id>[0-9]+)/results/$', views.results, name='results'),
    url(r'^(?P<qtweet_id>[0-9]+)/vote/$', views.vote, name='vote'),
	url(r'^load/$', views.load_tweets, name='load_tweets'),
]

"""
   
    """