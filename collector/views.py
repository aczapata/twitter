import time, datetime
import json
import re,string,operator
import math

from collections import Counter,defaultdict
from django.shortcuts import render,get_object_or_404,HttpResponse,HttpResponseRedirect
from .models import TwitterData,Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse,HttpResponseRedirect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams

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

positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
    # shall we also include game-specific terms?
    # 'triumph', 'triumphal', 'triumphant', 'victory', etc.
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
    # 'defeat', etc.
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
    count_all = Counter()
    count_hash = Counter()
    count_only = Counter()
    count_bigrams= Counter()
    count_single= Counter()
    count_stop_single= Counter()
    terms_max2=[]
    com = defaultdict(lambda : defaultdict(int))

    for tweet in tweets_list:
        # Create a list with all the terms
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'RT','via']

        terms_all = [term for term in preprocess(tweet.content)]
        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_only = [term for term in terms_all if term not in stop and not term.startswith(('#', '@'))] 
        terms_stop = [term for term in terms_all if term not in stop]
        terms_bigram = bigrams(terms_stop)
        terms_single = set(terms_all)
        terms_stop_single = set(terms_stop)

        for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
           
        com_max = []
        # For each term, look for the most common co-occurrent terms
        for t1 in com:
           t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
           for t2 in t1_max_terms:
               com_max.append(((t1, t2), com[t1][t2]))
                # Get the most frequent co-occurrences
     
        # Count terms only once, equivalent to Document Frequency
      
        count_all.update(terms_stop)
        count_hash.update(terms_hash)
        count_only.update(terms_only)
        count_bigrams.update(terms_bigram)
        count_single.update(terms_single)
        count_stop_single.update(terms_stop_single)   
    
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
    n_docs=95 
    for term, n in count_stop_single.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
           p_t_com[term][t2] = com[term][t2] / n_docs

    pmi = defaultdict(lambda : defaultdict(int))
    
    for t1 in p_t:
        for t2 in com[t1]:
            denom = p_t[t1] * p_t[t2]
            if denom != 0:
                pmi[t1][t2] = math.log((p_t_com[t1][t2] / denom ), 2)
     
    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc

    semantic_sorted = sorted(semantic_orientation.items(),key=operator.itemgetter(1),reverse=True)
    
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    terms_max2=terms_max[:10]
    count_all=count_all.most_common(10)
    count_hash=count_hash.most_common(10)
    count_only=count_only.most_common(10)   
    count_bigrams=count_bigrams.most_common(10) 
    context = {'tweets_list': tweets_list, 'count_all': count_all, 'count_only': count_only,'count_hash': count_hash,'count_bigrams': count_bigrams, 'terms_max': terms_max2, 'top_pos':top_pos, 'top_neg':top_neg}
    return render(request, 'collector/statistics.html', context)

def search (request):
    if ('q' in request.GET) and request.GET['q'].strip():
        query_string= request.GET['q']
        tweets_list = TwitterData.objects.all()
        count_search = Counter()
        for tweet in tweets_list:
            terms_only = [term for term in preprocess(tweet.content) 
                          if term not in stop 
                          and not term.startswith(('#', '@'))]
            if query_string in terms_only:
                count_search.update(terms_only)
        
        found_entries= count_search.most_common(20)
    
    return render_to_response('collector/statistics.html',
                          { 'query_string': query_string, 'found_entries': found_entries },
                          context_instance=RequestContext(request))

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
    terms_only = [term for term in terms_all if term not in stop and not term.startswith(('#', '@'))]    
    count_all = Counter()
    count_all.update(terms_stop)
    com = defaultdict(lambda : defaultdict(int))
    for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
           
    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
       t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
       for t2 in t1_max_terms:
           com_max.append(((t1, t2), com[t1][t2]))
            # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    terms_max = terms_max[:5]
    context= {'tweet_tokenized': tweet_tokenized, 'terms_stop': terms_stop,'tweet': tweet, 'count_all':count_all, 'terms_max': terms_max}
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

def graph(request):
    word_freq = count_terms_only.most_common(20)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('term_freq.json')