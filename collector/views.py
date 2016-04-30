import datetime
import json
import re
import string
import operator
import math
import nltk
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import sentlex
import sentlex.sentanalysis
import operator
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from django.shortcuts import render, get_object_or_404
from .models import TwitterData, Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.tag import pos_tag
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from .forms import UploadFileForm, SearchForm, FilterForm
from django.db.models import Q
from collector.tasks import load_file_task
from pattern.en import singularize

tls.set_credentials_file(username='melissaam', api_key='oghjiijwta')

# Variables
lemmatizer = WordNetLemmatizer()

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

word_features = []

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'RT', 'via']

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
    return render(request, 'collector/index2.html', context)

def list_tweets(request):
    tweets_list = TwitterData.objects.all()
    context = {'tweets_list': tweets_list}
    return render(request, 'collector/index.html', context)

def apply_filters(filters):
    filtered_data = Q()
    for t in filters:
        if filters[t]:
            if t == 'rp':
                filtered_data.add(Q(content__icontains="Republican"), Q.OR)
            if t == 'dp':
                filtered_data.add(Q(content__icontains="Democratic"), Q.OR)
            if t == 'dt':
                filtered_data.add(Q(content__icontains="Donald"), Q.OR)
                filtered_data.add(Q(content__icontains="Trump"), Q.OR)
            if t == 'hc':
                filtered_data.add(Q(content__icontains="Hillary"), Q.OR)
                filtered_data.add(Q(content__icontains="Clinton"), Q.OR)
            if t == 'tc':
                filtered_data.add(Q(content__icontains="Ted"), Q.OR)
                filtered_data.add(Q(content__icontains="Cruz"), Q.OR)
            if t == 'bs':
                filtered_data.add(Q(content__icontains="Bernie"), Q.OR)
                filtered_data.add(Q(content__icontains="Sanders"), Q.OR)
            if t == 'mr':
                filtered_data.add(Q(content__icontains="Marco"), Q.OR)
                filtered_data.add(Q(content__icontains="Rubio"), Q.OR)

    tweets_list = TwitterData.objects.filter(filtered_data)
    return tweets_list

def filter(request):
    if request.method == "POST":
        form = FilterForm(request.POST)
        if form.is_valid():
            # Parties
            filters = {}
            filters.setdefault('rp', form.cleaned_data['rp'])
            filters.setdefault('dp', form.cleaned_data['dp'])
            # Candidates
            filters.setdefault('dt', form.cleaned_data['dt'])
            filters.setdefault('hc', form.cleaned_data['hc'])
            filters.setdefault('tc', form.cleaned_data['tc'])
            filters.setdefault('mr', form.cleaned_data['mr'])
            filters.setdefault('bs', form.cleaned_data['bs'])

            # Events
            filters.setdefault('st', form.cleaned_data['st'])
            filters.setdefault('fp', form.cleaned_data['fp'])
            filters.setdefault('dd', form.cleaned_data['dd'])
            filters.setdefault('rd', form.cleaned_data['rd'])

            #analysis(apply_filters(filters))
            filled_form = FilterForm(request.POST)
            form = FilterForm()
            file = open("json_data", "r")
            json_data = file.readline()
            context = json.loads(json_data)
            context.update({'form': form , 'filled_form': filled_form})
            return render(request, 'collector/filter.html', context)
        else:
            print form.errors

    else:
        #analysis(TwitterData.objects.all())
        form = FilterForm()
        filled_form = form
        file = open("json_data", "r")
        json_data = file.readline()
        context = json.loads(json_data)
        context.update({'form': form, 'filled_form': filled_form})
    return render(request, 'collector/filter.html', context)

def tag_tweets(tweets_list):
    tagged_tweets = []
    not_tagged_tweets = []
    for tweet in tweets_list:
        if tweet.lang == 'en':
            dict_tagged_sentences = [
                tag for tag in tag_sentence(tweet, nltk.pos_tag(preprocess(tweet.content)))]
            if dict_tagged_sentences[1] != 'neutral':
                terms_stop = [term for term in preprocess(tweet.content) if term not in stop]
                t = (terms_stop, dict_tagged_sentences[1])
                tagged_tweets.append(t)
            else:
                not_tagged_tweets.append(tweet.content)
    return { 'tagged_tweets':tagged_tweets, 'not_tagged_tweets':not_tagged_tweets}

def analysis(tweets_list):

    count_hash = Counter()
    count_user = Counter()
    count_owner = Counter()

    terms_lang = []
    terms_owner = []
    lexicon_tag = []
    positive = 0
    negative = 0
    neutral = 0
    SWN = sentlex.SWN3Lexicon()
    classifier = sentlex.sentanalysis.BasicDocSentiScore()
    for tweet in tweets_list:
        # Create a list with all the terms
        terms_all = preprocess(tweet.content)
        terms_stop = [term for term in terms_all if term not in stop]

        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_user = [term for term in terms_all if term.startswith('@')]
        terms_owner.append(tweet.tweet_user)

        terms_positive = [term for term in terms_all if term in positive_vocab]
        terms_negative = [term for term in terms_all if term in negative_vocab]

        if tweet.lang == 'en':
            dict_tagged_sentences = [
                tag for tag in tag_sentence(tweet, nltk.pos_tag(preprocess(tweet.content)))]
            lexicon_tag.append(dict_tagged_sentences)

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

        count_hash.update(terms_hash)
        count_user.update(terms_user)

    print lexicon_tag
    y_axis_total = []

    y_axis_total.append(positive)
    y_axis_total.append(negative)
    y_axis_total.append(neutral)
    x_axis = ['positive', 'negative', 'neutral']

    x = list(Counter(terms_lang))
    y = Counter(terms_lang).values()

    count_owner.update(terms_owner)
    context_json = {
               'tweets_number': len(tweets_list),
               'hashtags_number': len(list(count_hash)),
               'users_number': len(list(count_user)),
               'owners_number': len(list(count_owner)),
               }
    file = open("json_data", "w")
    json.dump(context_json, file)

def tag_sentence(tweet, sentence, tag_with_lemmas=False):

    tag_sentence = []
    total_sentiment = 0
    N = len(sentence)
    i = 0
    while i < N:
        check = sentence[i][0]
        tagged_vocab_words = [j.split('\t')[0] for j in tagged_vocab]
        if sentence[i][0] == 'not' and i != N-1 :
            check = transform(sentence[i + 1][1], sentence[i + 1][0])
            if check in tagged_vocab_words:
                total_sentiment += float(
                    tagged_vocab[tagged_vocab_words.index(check)].split('\t')[1]) * -1
                i += 1
        else:
            check = transform(sentence[i][1], sentence[i][0])
            if check in tagged_vocab_words:
                total_sentiment += float(
                    tagged_vocab[tagged_vocab_words.index(check)].split('\t')[1])
        i += 1
    t= (tweet.content, sentiment(total_sentiment))
    return t


def sentiment(value):
    if value > 1.0:
        return "positive"
    elif value < -1.0:
        return "negative"
    else:
        return "neutral"


def transform(term, term_modified):
    if term == 'VBZ' or term == 'VBP' or term == 'VBN' or term == 'VBG' or term == 'VBD':
        return lemmatizer.lemmatize(''.join(term_modified), 'v')
    elif term == 'NNS':
        return singularize(''.join(term_modified))
    else:
        return term_modified


def tagged_words(terms):
    POS = [pos_tag(terms) for term in terms]
    POS = [[(word, word, [postag]) for (word, postag) in term] for term in POS]
    return POS


def tweets_tokenize(request):
    tweets_list = TwitterData.objects.all()
    form = SearchForm()
    analysis(tweets_list)
    file = open("json_data", "r")
    json_data = file.readline()
    context = json.loads(json_data)
    context.update({'form': form})
    return render(request, 'collector/filter.html', context)


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

def bayes_classifier(request):
    tagged_data = tag_tweets(TwitterData.objects.all())
    tweets_list = tagged_data['tagged_tweets']
    not_tagged_tweets_list = tagged_data['not_tagged_tweets']
    global word_features
    word_features = get_word_features(get_words_in_tweets(tweets_list))
    training_set = nltk.classify.apply_features(extract_features, tweets_list)
    tweet = tweets_list[0]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    for tweet in not_tagged_tweets_list:
        print classifier.classify(extract_features(tweet.split()))
    context = {'tweets_list': tweets_list, 'word_features': word_features, 'training_set': training_set}
    return render(request,'collector/bayes.html', context)


def train(labeled_featuresets, estimator=ELEProbDist):
    label_probdist = estimator(label_freqdist)
    feature_probdist = {}
    return NaiveBayesClassifier(label_probdist, feature_probdist)

def get_words_in_tweets(tweets_list):
    all_words = []
    for (words, sentiment) in tweets_list:
      all_words.extend(words)
    return all_words

def get_word_features(tweets_list):
    wordlist = nltk.FreqDist(tweets_list)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

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
    analyze_tweet = tag_sentence(tweet, nltk.pos_tag(preprocess(tweet.content.lower())))
    context = lexicon_tweet(tweet)
    context.update({'tweet': tweet, 'analyze_tweet': analyze_tweet[1]})
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
