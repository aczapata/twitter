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
#lombardy = '8.4931,44.9026,11.4316,46.6381'
#01/27/2016 Italia
def twitter(request, bbox='30,45,-125,-83'):
    N=2000  # tweets to request from each query
    S=200  # radius in miles
    lats=[38.9,40.7,37.8,39,37.4,28,30,42.4,48,36,32.3,33.5,34.7,33.8,37.2,41.2,46.8,
           46.6,37.2,43,42.7,40.8,36.2,38.6,35.8,40.3,43.6,40.8,44.9,44.9]
    
    lons=[-77,-74,-122,-105.5,-122,-82.5,-98,-71,-122,-115,-86.3,-112,-92.3,-84.4,-93.3,
           -104.8,-100.8,-112, -93.3,-89,-84.5,-111.8,-86.8,-92.2,-78.6,-76.8,-116.2,-98.7,-123,-93]
    
    #cities=DC,New York,San Fransisco,Colorado,Mountainview,Tampa,Austin,Boston,
    #       Seatle,Vegas,Montgomery,Phoenix,Little Rock,Atlanta,Springfield,
    #       Cheyenne,Bisruk,Helena,Springfield,Madison,Lansing,Salt Lake City,Nashville
    #       Jefferson City,Raleigh,Harrisburg,Boise,Lincoln,Salem,St. Paul
    
    url = "https://api.twitter.com/1.1/search/tweets.json?q=%23FreeKesha&result_type=popular"
    stream = oauth_req(url)
    
    
    for  line in stream:
        if line.endswith('\r\n'):
            try:
	             tweet = json.loads(line)
	             print 'Tweet: ' +str(tweet.get('user'))
	             
            except:
                print line

       
    return HttpResponse('Finished!\nlast response: </br>'+"<a href='"+str(stream)+"' >"+str(stream)+"</a>")
                  
    
"""   
def twitter(request, bbox='6.63,36.46,18.78,47.09'):

    url = "https://stream.twitter.com/1.1/statuses/filter.json?locations=%s" % (bbox)
    stream = oauth_req(url)
    for  line in stream:
        if line.endswith('\r\n'):
            try:
	             tweet = json.loads(line)
	             if tweet.get('coordinates'):
	                tweet_db = TwitterData( latitude=tweet['coordinates']['coordinates'][1], longitude=tweet['coordinates']['coordinates'][0],
	                                        user=tweet['user']['screen_name'], date=datetime.datetime.strptime(str(tweet['created_at']), "%a %b %d %H:%M:%S +%f %Y"))
	                #print tweet_db
	                if tweet.get('source'):
	                    tweet_db.source = tweet['source']
	                if tweet['user'].get('location'):
	                    tweet_db.user_location = tweet['user']['location']
	                tweet_db.save()
	                print 'hay coordenadas! ' + str(tweet['coordinates'])
	                 #else:
	                    #print tweet
            except:
                print line

        
    return HttpResponse('Finished!\nlast response: </br>'+"<a href='"+str(stream)+"' >"+str(stream)+"</a>")

"""

def oauth_req(url, http_method="GET", post_body="", http_headers=None):
    consumer = oauth.Consumer(key=settings.TWITTER_API_KEY, secret=settings.TWITTER_CONSUMER_SECRET)
    token = oauth.Token(key='4730844797-Mx4ZnMtKeKFc4Osgah6uzxgeNjkTL0UlhL1rVPA', secret='lqAPIEyRQiLpxctE2MQ7ruqjIS0Za7AiabJC3ikpdQYnq')
    client = oauth.Client(consumer, token)
    params = {
    'oauth_version': "1.0",
    'oauth_nonce': oauth.generate_nonce(),
    'oauth_timestamp': str(int(time.time())),
    }
    params['oauth_token'] = token.key
    params['oauth_consumer_key'] = consumer.key
    req = oauth.Request(method=http_method, url=url, parameters=params)
    signature_method = oauth.SignatureMethod_HMAC_SHA1()
    req.sign_request(signature_method, consumer, token)
    req.to_url()
    #resp, content = client.request(req.to_url(), method=http_method, body=post_body, headers=http_headers )
    #print resp
    #print content
    rs = urllib2.urlopen(req.to_url())
    #print rs
    return rs