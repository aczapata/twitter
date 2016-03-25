import time, datetime
import json
import re,string,operator

from collections import Counter
from django.shortcuts import render,get_object_or_404,HttpResponse,HttpResponseRedirect
from .models import TwitterData,Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse,HttpResponseRedirect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Variables
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


# Methods.
def index(request):
    tweets_list = TwitterData.objects.all()
    context = {'tweets_list': tweets_list}
    return render(request, 'collector/index.html', context)

def tweets_tokenize(request):
    tweets_list = TwitterData.objects.all()
    for tweet in tweets_list:
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet.content)]
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'via']
        terms_stop = [term for term in terms_all if term not in stop]
        count_all = Counter()
        count_all.update(terms_stop)
        # Update the counter
        count_all.update(terms_all)
    common_terms = count_all.most_common(10)
    context = {'tweets_list': tweets_list, 'count_all': count_all, 'common_terms': common_terms}
    return render(request, 'collector/statistics.html', context)

def detail(request, tweet_id):
    tweet= get_object_or_404(TwitterData, pk=tweet_id)
    return render(request, 'collector/detail.html', {'tweet': tweet})

def vote(request, tweet_id):
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    try:
        selected_choice = tweet.sentiment_set.get(pk=request.POST['choice'])
    except (KeyError, Sentiment.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'collector/detail.html', {
            'tweet': tweet,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()

        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('collector:results', args=(tweet.id,)))

def results(request, tweet_id):
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    return render(request, 'collector/results.html', {'tweet': tweet})

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def tweet_tokenize(request,tweet_id):
    
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    tweet_tokenized=preprocess(tweet.content)
    terms_all = [term for term in tweet_tokenized]
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']
    terms_stop = [term for term in terms_all if term not in stop]
    count_all = Counter()
    count_all.update(terms_stop)
    context= {'tweet_tokenized': tweet_tokenized, 'terms_stop': terms_stop,'tweet': tweet, 'count_all':count_all}
    return render(request, 'collector/tokenize.html', context)
 

def load_tweets(request):
    tweets_data_path = '../TwitterData240320161624.txt'
    tweets_file = open(tweets_data_path, "r")
    count=0
    for line in tweets_file:
        try:
            tweet = json.loads(line)

            tweet_db = TwitterData( tweet_id= tweet['id_str'],content= tweet['text'],user=tweet['user']['screen_name'], date=datetime.datetime.strptime(str(tweet['created_at']), "%a %b %d %H:%M:%S +%f %Y"))
            if tweet.get('coordinates'):
                tweet_db.latitude=tweet['coordinates']['coordinates'][1]
                tweet_db.longitude=tweet['coordinates']['coordinates'][0]
            if tweet.get('source'):
               tweet_db.source = tweet['source']
            if tweet['user'].get('location'):
               tweet_db.user_location = tweet['user']['location']
            if tweet.get('lang'):
               tweet_db.source = tweet['lang']
            tweet_db.save()
            count=count+1
            tweet_db.sentiment_set.create(sentiment_text="IRR")
            tweet_db.sentiment_set.create(sentiment_text="UND")
            tweet_db.sentiment_set.create(sentiment_text="NEG")
            tweet_db.sentiment_set.create(sentiment_text="POS")
        except:
            continue
    return HttpResponse('Finished!\n' +str(count)+" Tweets loaded")