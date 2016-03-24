import time
import datetime
import json
import urllib2
import oauth2 as oauth
from twitter import settings
from django.shortcuts import render
from models import *
from django.http import HttpResponse

# Create your views here.

def twitter(request):
    tweets_data_path = '../TwitterData010320161441.txt'
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)

            tweet_db = TwitterData( tweet_id= tweet['id_str'], content = tweet['text'], user=tweet['user']['screen_name'], date=datetime.datetime.strptime(str(tweet['created_at']), "%a %b %d %H:%M:%S +%f %Y"))
            
            if tweet.get('coordinates'):
                tweet_db.latitude=tweet['coordinates']['coordinates'][1]
                tweet_db.longitude=tweet['coordinates']['coordinates'][0]
            if tweet.get('source'):
               tweet_db.source = tweet['source']
            if tweet['user'].get('location'):
               tweet_db.user_location = tweet['user']['location']
            if tweet['lang']:
                tweet_db.lang=tweet['lang']
            
            tweet_db.save()
        except:
            continue