from django.conf.urls import url
from . import views

app_name="collector"
urlpatterns = [
    # Examples:
    url(r'^$', views.index, name='index'),
    url(r'^load/$', views.upload_file, name='load_tweets'),
    url(r'^load/upload_file$', views.upload_file, name='load_tweets'),
    url(r'^statistics/$', views.tweets_tokenize, name='statistics'),
    url(r'^(?P<tweet_id>[0-9]+)/$', views.detail, name='detail'),
    url(r'^(?P<tweet_id>[0-9]+)/results/$', views.results, name='results'),
    url(r'^(?P<tweet_id>[0-9]+)/tokenize/$', views.tweet_tokenize, name='tokenize'),
    url(r'^(?P<tweet_id>[0-9]+)/vote/$', views.vote, name='vote'),
	
]
