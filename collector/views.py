import datetime
import json
import re
import string
import operator
import math
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import sentlex
import sentlex.sentanalysis
from collections import Counter, defaultdict
from django.shortcuts import render, get_object_or_404
from .models import TwitterData, Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.tag import pos_tag
from .forms import UploadFileForm, SearchForm
from collector.tasks import load_file_task

tls.set_credentials_file(username='melissaam', api_key='oghjiijwta')

# Variables
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    # URLs
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

positive_vocab = []
negative_vocab = []
tagged_vocab = []



def load_words(file, vector=[]):
    f = open(file, 'r')
    for line in f:
        vector.append(line.split('\n')[0])

load_words("./static/positive-words.txt", positive_vocab)
load_words("./static/negative-words.txt", negative_vocab)
load_words("./static/tagged-words.txt", tagged_vocab)


tokens_re = re.compile(
    r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(
    r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def index(request):
    tweets_list = TwitterData.objects.all()
    context = {'tweets_list': tweets_list}
    return render(request, 'collector/index.html', context)


def analysis(tweets_list):
    count_all = Counter()
    count_hash = Counter()
    count_user = Counter()
    count_only = Counter()
    count_bigrams = Counter()
    count_single = Counter()
    count_stop_single = Counter()
    count_positive = Counter()
    count_negative = Counter()
    com = defaultdict(lambda: defaultdict(int))
    terms_lang = []
    positive = 0
    negative = 0
    neutral = 0
    SWN = sentlex.SWN3Lexicon()
    classifier = sentlex.sentanalysis.BasicDocSentiScore()

    for tweet in tweets_list:
        # Create a list with all the terms
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'RT', 'via']

        terms_all = [term for term in preprocess(tweet.content)]
        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_user = [term for term in terms_all if term.startswith('@')]
        terms_stop = [term for term in terms_all if term not in stop]
        terms_positive = [term for term in terms_all if term in positive_vocab]
        terms_negative = [term for term in terms_all if term in negative_vocab]

        if tweet.lang is not None:
            terms_lang.append(tweet.lang)

        classifier.classify_document(
            tweet.content, tagged=False, L=SWN, a=True, v=True, n=True, r=False, negation=True, verbose=False)
        results = classifier.resultdata
        results_pos = results['resultpos']
        results_neg = results['resultneg']
        dif = abs(results_pos - results_neg)

        if(dif > 0.02):
            if(results_pos > results_neg):
                positive += 1
            else:
                negative += 1
        else:
            neutral += 1

        terms_bigram = bigrams(terms_stop)
        terms_stop_single = set(terms_stop)

        count_all.update(terms_stop)
        count_hash.update(terms_hash)
        count_user.update(terms_user)
        count_bigrams.update(terms_bigram)
        count_stop_single.update(terms_stop_single)
        count_positive.update(terms_positive)
        count_negative.update(terms_negative)

        for i in range(len(terms_stop) - 1):
            for j in range(i + 1, len(terms_stop)):
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
    p_t_com = defaultdict(lambda: defaultdict(int))
    n_docs = 95.0
    for term, n in count_stop_single.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    p_t_max_terms = max(p_t.items(), key=operator.itemgetter(1))[:5]

    pmi = defaultdict(lambda: defaultdict(int))

    for t1 in p_t:
        for t2 in com[t1]:
            try:
                denom = p_t[t1] * p_t[t2]
                if denom != 0:
                    pmi[t1][t2] = math.log((p_t_com[t1][t2] / denom), 2)
            except:
                continue
    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc

    semantic_sorted = sorted(
        semantic_orientation.items(), key=operator.itemgetter(1), reverse=True)

    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]

    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)[:10]
    count_all = count_all.most_common(10)
    count_hash = count_hash.most_common(10)
    count_user = count_user.most_common(10)
    count_only = count_only.most_common(10)
    count_single = count_single.most_common(10)
    count_stop_single = count_stop_single.most_common(10)
    count_bigrams = count_bigrams.most_common(10)

    y_axis_total = []

    y_axis_total.append(positive)
    y_axis_total.append(negative)
    y_axis_total.append(neutral)

    x_axis = ['positive', 'negative', 'neutral']
    plot_sen = graph_plot(x_axis, y_axis_total, "Sentiment", 'pie')

    x = list(Counter(terms_lang))
    y = Counter(terms_lang).values()
    plot_lan = graph_plot(x, y, "Languages", 'bar')
    context = {'tweets_list': tweets_list,
               'plot_sen': plot_sen, 'plot_lan': plot_lan,
               'count_all': count_all,
               'count_hash': count_hash, 'count_user': count_user,
               'count_bigrams': count_bigrams,
               'count_stop_single': count_stop_single,
               'terms_max': terms_max, 'p_t_max_terms': p_t_max_terms,
               'top_pos': top_pos, 'top_neg': top_neg}
    return context


def tagged_words(terms):
    POS = [pos_tag(terms) for term in terms]
    POS = [[(word, word, [postag]) for (word, postag) in term] for term in POS]

    return POS


def tweets_tokenize(request):

    tweets_list = TwitterData.objects.all()
    form = SearchForm
    context = analysis(tweets_list)
    context.update({'form': form})
    return render(request, 'collector/statistics.html', context)


def topic_filter(request):
    query = request.POST['text']
    return HttpResponseRedirect(
        reverse('collector:ind_statistics', args=(query,)))


def topic_tokenize(request, query):
    tweets_list = TwitterData.objects.filter(content__icontains=query)
    context = analysis(tweets_list)
    return render(request, 'collector/individual_statistics.html', context)


def geo(request):
    tweets_list = TwitterData.objects.filter(
        latitude__isnull=False, longitude__isnull=False)
    context = {'tweets_list': tweets_list}
    return render(request, 'collector/geo.html', context)


def graph_plot(x_axis, y_axis, name, graph_type):
    # Using plotly to graph with x,y parameter bar diagram
    if(graph_type is 'pie'):
        trace = dict(labels=x_axis, values=y_axis)
        data = [
            go.Pie(
                trace
            )
        ]
    else:
        trace = dict(x=x_axis, y=y_axis)
        data = [
            go.Bar(
                trace
            )
        ]
    plot = py.plot(data, filename=name, auto_open=False)
    return plot


def detail(request, tweet_id):
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
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
        return HttpResponseRedirect(
            reverse('collector:results', args=(tweet.id,)))


def results(request, tweet_id):
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    return render(request, 'collector/results.html', {'tweet': tweet})


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(
            token) else token.lower() for token in tokens]
    return tokens


def lexicon_tweet(tweet):
    SWN = sentlex.SWN3Lexicon()
    classifier = sentlex.sentanalysis.BasicDocSentiScore()
    classifier.classify_document(
        tweet.content, tagged=False, L=SWN, a=True, v=True, n=True, r=False, negation=True, verbose=False)
    results = classifier.resultdata
    results_pos = results['resultpos']
    results_neg = results['resultneg']
    dif = abs(results_pos - results_neg)
    context = {'results': results, 'dif': dif,
               'results_pos': results_pos, 'results_neg': results_neg}
    return context


def tweet_tokenize(request, tweet_id):

    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    context = lexicon_tweet(tweet)
    context.update({'tweet': tweet})
    return render(request, 'collector/tokenize.html', context)


def load_tweets(tweets_file):

    count = 0
    for line in tweets_file:
        try:
            tweet = json.loads(line)

            tweet_db = TwitterData(tweet_id=tweet['id_str'],
                                   content=tweet['text'],
                                   user=tweet['user']['screen_name'],
                                   date=datetime.datetime.strptime(
                str(tweet['created_at']), "%a %b %d %H:%M:%S +%f %Y"))
            if tweet.get('coordinates'):
                tweet_db.latitude = tweet['coordinates']['coordinates'][1]
                tweet_db.longitude = tweet['coordinates']['coordinates'][0]
            if tweet.get('source'):
                tweet_db.source = tweet['source']
            if tweet['user'].get('location'):
                tweet_db.user_location = tweet['user']['location']
            if tweet['lang']:
                tweet_db.lang = tweet['lang']

            tweet_db.save()
            count = count + 1
            tweet_db.sentiment_set.create(sentiment_text="IRR")
            tweet_db.sentiment_set.create(sentiment_text="UND")
            tweet_db.sentiment_set.create(sentiment_text="NEG")
            tweet_db.sentiment_set.create(sentiment_text="POS")
        except:
            continue

    return HttpResponse('Finished!\n' + str(count) + " Tweets loaded")


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        number = TwitterData.objects.count()
        if form.is_valid():
            tweets_file = form.save()
            load_file_task.delay(tweets_file)
            message = "Loaded file!"
            context = {'form': form, 'number': number, 'message': message}
            return render(request, 'collector/upload.html', context)
        else:
            message = "Ups Failed!"
            context = {'form': form, 'number': number, 'message': message}
            return render(request, 'collector/upload.html', context)
    else:
        number = TwitterData.objects.count()
        form = UploadFileForm()
        context = {'form': form, 'number': number}
        return render(request, 'collector/upload.html', context)
