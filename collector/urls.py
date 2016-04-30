from django.conf.urls import url
from . import views

app_name = "collector"
urlpatterns = [
    # Examples:
    url(r'^$', views.index, name='index'),
    url(r'^list/$', views.list_tweets, name='list'),
    url(r'^bayes/$', views.bayes_classifier, name='bayes'),
    url(r'^load/$', views.upload_file, name='load_tweets'),
    url(r'^load/upload_file$', views.upload_file, name='load_tweets'),
    url(r'^statistics/$', views.tweets_tokenize, name='statistics'),
    url(r'^filter/$', views.filter, name='filter'),
    url(r'^statistics/geo/$', views.geo, name='geo'),
    url(r'^statistics/topic_tokenize/$', views.topic_filter, name='topic_filter'),
    url(r'^statistics/(?P<query>\D+)/$', views.topic_tokenize, name='ind_statistics'),
    url(r'^(?P<tweet_id>[0-9]+)/$', views.detail, name='detail'),
    url(r'^(?P<tweet_id>[0-9]+)/results/$', views.results, name='results'),
    url(r'^(?P<tweet_id>[0-9]+)/tokenize/$', views.tweet_tokenize, name='tokenize'),
    url(r'^(?P<tweet_id>[0-9]+)/vote/$', views.vote, name='vote'),
]
