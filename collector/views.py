import time, datetime
import json
import re,string,operator
import math
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter,defaultdict
from django.shortcuts import render,get_object_or_404,HttpResponse,HttpResponseRedirect
from .models import TwitterData,Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse,HttpResponseRedirect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams

from .forms import UploadFileForm, SearchForm
tls.set_credentials_file(username='aczapata', api_key='4x5zrxkr6n')

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
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(', 'racist'
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
    form = SearchForm
    count_all = Counter()
    count_hash = Counter()
    count_user = Counter()
    count_only = Counter()
    count_bigrams= Counter()
    count_single= Counter()
    count_stop_single= Counter()
    terms_max2=[]
    com = defaultdict(lambda : defaultdict(int))
    terms_lang = []

    for tweet in tweets_list:
        # Create a list with all the terms
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'RT','via']

        terms_all = [term for term in preprocess(tweet.content)]
        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_user = [term for term in terms_all if term.startswith('@')]
        terms_stop = [term for term in terms_all if term not in stop]
        terms_only = [term for term in terms_all if term not in stop and not term.startswith(('#', '@'))] 
        terms_lang.append(tweet.lang)
        

        terms_bigram = bigrams(terms_stop)
        
        terms_single = set(terms_all)
        terms_stop_single = set(terms_stop)

        count_all.update(terms_stop)
        count_hash.update(terms_hash)
        count_user.update(terms_user)
        count_only.update(terms_only)
        count_bigrams.update(terms_bigram)
        count_single.update(terms_single)
        count_stop_single.update(terms_stop_single)

        for i in range(len(terms_stop)-1):            
           for j in range(i+1, len(terms_stop)):
               w1, w2 = sorted([terms_stop[i], terms_stop[j]])                
               if w1 != w2:
                   com[w1][w2] += 1
                   com[w2][w1] += 1
               
        com_max = []
            # For each term, look for the most common co-occurrent terms
        for t1 in com:
           t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
           for t2 in t1_max_terms:
               com_max.append(((t1, t2), com[t1][t2]))

    
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
    n_docs=95.0 
    for term, n in count_stop_single.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    p_t_max_terms = max(p_t.items(), key=operator.itemgetter(1))[:5]
    
    pmi = defaultdict(lambda : defaultdict(int))

    for t1 in p_t:
        for t2 in com[t1]:
            try:
               denom = p_t[t1] * p_t[t2]
               if denom != 0:
                    pmi[t1][t2] = math.log((p_t_com[t1][t2] / denom ), 2)
            except:
               continue

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
    count_user=count_user.most_common(10)
    count_only=count_only.most_common(10)
    count_single=count_single.most_common(10)
    count_stop_single=count_stop_single.most_common(10)    
    count_bigrams=count_bigrams.most_common(10) 
    """
    y_axis_pos = []
    y_axis_neg = []
    y_axis_total = []
    for n1,n2 in top_pos:
        y_axis_pos.append(n2)

    y_axis_total.append((sum(y_axis_pos))/len(y_axis_pos))

    for n1,n2 in top_neg:
        y_axis_neg.append(n2)
    
    y_axis_total.append(abs((sum(y_axis_neg))/len(y_axis_neg)))
    x_axis = ['positive', 'negative']  
    x=list(Counter(terms_lang))
    y=Counter(terms_lang).values()
    plot = graph_plot(x_axis,y_axis_total, "Sentiment")
    #para graficar idiomas enviamos x,y simplemente no lo puse en el html para ordenarlo despues pero sirve dont worry
    context = {'tweets_list': tweets_list, 'form':form,'plot': plot, 'count_all': count_all, 'count_only': count_only,'count_hash': count_hash,'count_user': count_user,'count_bigrams': count_bigrams, 'count_single': count_single,'count_stop_single': count_stop_single, 'terms_max': terms_max2,'p_t_max_terms': p_t_max_terms, 'top_pos':top_pos, 'top_neg':top_neg}
    """
    context = {'tweets_list': tweets_list, 'form':form, 'count_all': count_all, 'count_only': count_only,'count_hash': count_hash,'count_user': count_user,'count_bigrams': count_bigrams, 'count_single': count_single,'count_stop_single': count_stop_single, 'terms_max': terms_max2,'p_t_max_terms': p_t_max_terms, 'top_pos':top_pos, 'top_neg':top_neg}
    return render(request, 'collector/statistics.html', context)

def topic_filter(request):
    form = SearchForm(request.POST)
    query= request.POST['text']
    return HttpResponseRedirect(reverse('collector:ind_statistics', args=(query,)))
    
def topic_tokenize(request, query):
    tweets_list = TwitterData.objects.filter(content__icontains= query)
    form = SearchForm
    count_all = Counter()
    count_hash = Counter()
    count_user = Counter()
    count_only = Counter()
    count_bigrams= Counter()
    count_single= Counter()
    count_stop_single= Counter()
    terms_max2=[]
    com = defaultdict(lambda : defaultdict(int))
    terms_lang = []

    for tweet in tweets_list:
        # Create a list with all the terms
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'RT','via']

        terms_all = [term for term in preprocess(tweet.content)]
        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_user = [term for term in terms_all if term.startswith('@')]
        terms_stop = [term for term in terms_all if term not in stop]
        terms_only = [term for term in terms_all if term not in stop and not term.startswith(('#', '@'))] 
        terms_lang.append(tweet.lang)
        

        terms_bigram = bigrams(terms_stop)
        
        terms_single = set(terms_all)
        terms_stop_single = set(terms_stop)

        count_all.update(terms_stop)
        count_hash.update(terms_hash)
        count_user.update(terms_user)
        count_only.update(terms_only)
        count_bigrams.update(terms_bigram)
        count_single.update(terms_single)
        count_stop_single.update(terms_stop_single)

        for i in range(len(terms_stop)-1):            
           for j in range(i+1, len(terms_stop)):
               w1, w2 = sorted([terms_stop[i], terms_stop[j]])                
               if w1 != w2:
                   com[w1][w2] += 1
                   com[w2][w1] += 1
               
        com_max = []
            # For each term, look for the most common co-occurrent terms
        for t1 in com:
           t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
           for t2 in t1_max_terms:
               com_max.append(((t1, t2), com[t1][t2]))

    
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
    n_docs=95.0 
    for term, n in count_stop_single.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    p_t_max_terms = max(p_t.items(), key=operator.itemgetter(1))[:5]
    
    pmi = defaultdict(lambda : defaultdict(int))

    for t1 in p_t:
        for t2 in com[t1]:
            try:
               denom = p_t[t1] * p_t[t2]
               if denom != 0:
                    pmi[t1][t2] = math.log((p_t_com[t1][t2] / denom ), 2)
            except:
               continue

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
    count_user=count_user.most_common(10)
    count_only=count_only.most_common(10)
    count_single=count_single.most_common(10)
    count_stop_single=count_stop_single.most_common(10)    
    count_bigrams=count_bigrams.most_common(10) 
    """
    y_axis_pos = []
    y_axis_neg = []
    y_axis_total = []
    for n1,n2 in top_pos:
        y_axis_pos.append(n2)

    y_axis_total.append((sum(y_axis_pos))/len(y_axis_pos))

    for n1,n2 in top_neg:
        y_axis_neg.append(n2)
    
    y_axis_total.append(abs((sum(y_axis_neg))/len(y_axis_neg)))
    x_axis = ['positive', 'negative']  
    x=list(Counter(terms_lang))
    y=Counter(terms_lang).values()
    plot = graph_plot(x_axis,y_axis_total, "Sentiment")
    #para graficar idiomas enviamos x,y simplemente no lo puse en el html para ordenarlo despues pero sirve dont worry
    context = {'tweets_list': tweets_list, 'form':form,'plot': plot, 'count_all': count_all, 'count_only': count_only,'count_hash': count_hash,'count_user': count_user,'count_bigrams': count_bigrams, 'count_single': count_single,'count_stop_single': count_stop_single, 'terms_max': terms_max2,'p_t_max_terms': p_t_max_terms, 'top_pos':top_pos, 'top_neg':top_neg}
    """
    context = {'tweets_list': tweets_list, 'form':form, 'count_all': count_all, 'count_only': count_only,'count_hash': count_hash,'count_user': count_user,'count_bigrams': count_bigrams, 'count_single': count_single,'count_stop_single': count_stop_single, 'terms_max': terms_max2,'p_t_max_terms': p_t_max_terms, 'top_pos':top_pos, 'top_neg':top_neg}
    return render(request, 'collector/individual_statistics.html', context)

def graph_plot(x_axis,y_axis, name):
   #Using plotly to graph with x,y parameter bar diagram
    trace = dict(x=x_axis, y=y_axis)
    data = [
        go.Bar(
            trace
        )
    ]
    plot = py.plot(data, filename=name, auto_open=False)
    return plot    

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

    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'RT','via']

    terms_all = [term for term in preprocess(tweet.content)]
    terms_hash = [term for term in terms_all if term.startswith('#')]
    terms_user = [term for term in terms_all if term.startswith('@')]
    terms_stop = [term for term in terms_all if term not in stop]
    terms_only = [term for term in terms_all if term not in stop and not term.startswith(('#', '@'))] 
        

    terms_bigram = bigrams(terms_stop)
        
    terms_single = set(terms_all)
    terms_stop_single = set(terms_stop)

    com = defaultdict(lambda : defaultdict(int))
    for i in range(len(terms_stop)-1):            
       for j in range(i+1, len(terms_stop)):
           w1, w2 = sorted([terms_stop[i], terms_stop[j]])                
           if w1 != w2:
               com[w1][w2] += 1
               
    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in terms_stop:
        try:
            for t2 in com[t1]:
                print (t1+","+t2 +":")
        except:
            continue
    
    """        
    for t1 in com:
       t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
       for t2 in t1_max_terms:
           com_max.append(((t1, t2), com[t1][t2]))
    """
    context= {'tweet': tweet,'terms_stop': terms_stop,'terms_only': terms_only,'terms_single': terms_single,'terms_stop_single': terms_stop_single, 'com': com, 'com_max': com_max}
    return render(request, 'collector/tokenize.html', context)

def load_tweets(tweets_file):
    
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
            if tweet['lang']:
                tweet_db.lang=tweet['lang']
            
            tweet_db.save()
            count=count+1
            tweet_db.sentiment_set.create(sentiment_text="IRR")
            tweet_db.sentiment_set.create(sentiment_text="UND")
            tweet_db.sentiment_set.create(sentiment_text="NEG")
            tweet_db.sentiment_set.create(sentiment_text="POS")
        except:
            continue

    return HttpResponse('Finished!\n' +str(count)+" Tweets loaded")

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        number= TwitterData.objects.count()
        if form.is_valid():
            load_tweets(request.FILES['file'])
            message= "Loaded file!"
            return render(request, 'collector/upload.html', {'form': form, 'number':number, 'message':message, 'number':number})
        else:
            message= "Error with file!"
            return render(request, 'collector/upload.html', {'form': form, 'number':number, 'message':message, 'number':number})
         
    else:
        number= TwitterData.objects.count()
        form = UploadFileForm()
    return render(request, 'collector/upload.html', {'form': form, 'number':number})
