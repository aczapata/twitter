import datetime
import json
import re
import string
import nltk
import sentlex
import sentlex.sentanalysis
import random
from nltk.stem import WordNetLemmatizer
from collections import Counter
from django.shortcuts import render, get_object_or_404
from .models import TwitterData, Sentiment
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.probability import ELEProbDist
from nltk import NaiveBayesClassifier
from .forms import UploadFileForm, SearchForm, FilterForm, CompareForm
from django.db.models import Q
from collector.tasks import load_file_task
from pattern.en import singularize
from nvd3 import pieChart


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


tokens_re = re.compile(
    r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(
    r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def load_words(file, vector=[]):
    f = open(file, 'r')
    for line in f:
        vector.append(line.split('\n')[0])

load_words("./static/positive-words.txt", positive_vocab)
load_words("./static/negative-words.txt", negative_vocab)
load_words("./static/tagged-words.txt", tagged_vocab)


def index(request):
    return render(request, 'collector/index.html')


def list_tweets(request):
    tweets_list = TwitterData.objects.all()
    context = {'tweets_list': tweets_list}
    return render(request, 'collector/list.html', context)


def function_to_graph(x_axis, y_axis, title):
    chart = pieChart(name=title, color_category='category20c', height=450, width=450)

    xdata = x_axis
    ydata = y_axis

    extra_serie = {"tooltip": {"y_start": "", "y_end": "tweets"}}
    chart.add_serie(y=ydata, x=xdata, extra=extra_serie)
    chart.buildcontent()
    return chart.htmlcontent


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
    analysis(TwitterData.objects.all())
    form = FilterForm()
    compare_form = CompareForm()
    file = open("json_data", "r")
    json_data = file.readline()
    context = json.loads(json_data)

    x = context['x_axis_language']
    y = context['y_axis_language']
    graph_languages = function_to_graph(x, y, 'language')

    x1 = context['x_axis_sentiment']
    y1 = context['y_axis_sentiment']
    graph_sentiment = function_to_graph(x1, y1, 'sentiment')

    if request.method == "POST":
        if 'compare' in request.POST:
            compare_form = CompareForm(request.POST)
            if compare_form.is_valid():
                option1 = compare_form.cleaned_data['option1']
                option2 = compare_form.cleaned_data['option2']
                event = compare_form.cleaned_data['event']
                """
                options.setdefault('option1', compare_form.cleaned_data['option1'])
                options.setdefault('option2', compare_form.cleaned_data['option2'])
                options.setdefault('event', compare_form.cleaned_data['event'])
                """
                return HttpResponseRedirect(reverse('collector:compare', args=(option1, option2, event)))
        elif 'filter' in request.POST:
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
                context.update({'form': form , 'filled_form': filled_form, 'compare_form': compare_form, 'graph_languages': graph_languages, 'graph_sentiment': graph_sentiment})
                return render(request, 'collector/filter.html', context)
            else:
                print form.errors
    else:

        filled_form = form
        context.update({'form': form , 'filled_form': filled_form, 'compare_form': compare_form,  'graph_languages': graph_languages, 'graph_sentiment': graph_sentiment})
    return render(request, 'collector/filter.html', context)


def tag_tweets_sentiment():
    tweets_list = TwitterData.objects.all()
    training_set = []
    classify_set = []
    for tweet in tweets_list:
        if tweet.lang == 'en':
            sentiment1 = tag_sentence(nltk.pos_tag(preprocess(tweet.content.lower())))
            sentiment2 = lexicon_tweet(tweet.content.lower())
            if sentiment1 == sentiment2:
                tweet.tweet_sentiment = sentiment1
                tweet.save()
                terms_stop = [term for term in preprocess(tweet.content.lower()) if term not in stop]
                t = (terms_stop, sentiment1)
                training_set.append(t)
            else:
                classify_set.append(tweet)
    bayes_classifier(training_set, classify_set)


def tag_tweet_sentiment(tweet):
    if tweet.lang == 'en':
            sentiment1 = tag_sentence(nltk.pos_tag(preprocess(tweet.content.lower())))
            sentiment2 = lexicon_tweet(tweet.content.lower())
            if sentiment1 == sentiment2:
                tweet.tweet_sentiment = sentiment1
                return tweet.tweet_sentiment
            else:
                return 'not_tagged'
    else:
        return 'no_english'


def analysis(tweets_list):

    count_hash = Counter()
    count_user = Counter()
    count_owner = Counter()

    terms_lang = []
    terms_owner = []
    terms_sentiment = []

    # tag_tweets_sentiment()
    for tweet in tweets_list:
        # Create a list with all the terms
        terms_all = preprocess(tweet.content)

        terms_hash = [term for term in terms_all if term.startswith('#')]
        terms_user = [term for term in terms_all if term.startswith('@')]
        terms_owner.append(tweet.tweet_user)

        if tweet.lang is not None:
            terms_lang.append(tweet.lang)

        if tweet.lang == 'en':
            terms_sentiment.append(tweet.tweet_sentiment)

        count_hash.update(terms_hash)
        count_user.update(terms_user)

    count_owner.update(terms_owner)

    x_lang = list(Counter(terms_lang))
    y_lang = Counter(terms_lang).values()

    x_sent = list(Counter(terms_sentiment))
    y_sent = Counter(terms_sentiment).values()

    context_json = {
                'tweets_number': len(tweets_list),
                'hashtags_number': len(list(count_hash)),
                'users_number': len(list(count_user)),
                'owners_number': len(list(count_owner)),
                'x_axis_language': x_lang,
                'y_axis_language': y_lang,
                'x_axis_sentiment': x_sent,
                'y_axis_sentiment': y_sent,
                'top_hashtags': count_hash.most_common(5),
                'top_users': count_user.most_common(5),
                'top_owners': count_owner.most_common(5),
                }
    file = open("json_data", "w")
    json.dump(context_json, file)


def tag_sentence(sentence, tag_with_lemmas=False):
    total_sentiment = 0
    N = len(sentence)
    i = 0
    while i < N:
        check = sentence[i][0]
        tagged_vocab_words = [j.split('\t')[0] for j in tagged_vocab]
        if sentence[i][0] == 'not' and i != N - 1:
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
    return sentiment(total_sentiment)


def sentiment(value):
    if value == 0:
        return 'irrelevant'
    else:
        if value > 1.0:
            return 'positive'
        elif value < -1.0:
            return 'negative'
        else:
            return 'neutral'


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


def lexicon_tweet(tweet):
    SWN = sentlex.SWN3Lexicon()
    classifier = sentlex.sentanalysis.BasicDocSentiScore()
    classifier.classify_document(
        tweet, tagged=False, L=SWN, a=True, v=True, n=True, r=False, negation=True, verbose=False)
    results = classifier.resultdata
    results_pos = results['resultpos']
    results_neg = results['resultneg']

    if results_pos == 0 and results_neg == 0:
        sentiment = 'irrelevant'
    else:
        dif = abs(results_pos - results_neg)
        if dif < 0.05:
            sentiment = 'neutral'
        else:
            if results_pos > results_neg:
                sentiment = 'positive'
            else:
                sentiment = 'negative'
    return sentiment


def tweets_tokenize(request):
    tweets_list = TwitterData.objects.all()
    form = SearchForm()
    analysis(tweets_list)
    file = open("json_data", "r")
    json_data = file.readline()
    context = json.loads(json_data)
    context.update({'form': form})
    return render(request, 'collector/filter.html', context)


def geo(request):
    tweets_list = TwitterData.objects.filter(
        latitude__isnull=False, longitude__isnull=False)
    geo_data=[]
    for tweet in tweets_list:
        geo_json_feature = {
           "type": "Feature",
           "properties": {
                "text": tweet.content,
            },
           "geometry": {
                "type": "Point",
                "coordinates": [float(tweet.longitude), float(tweet.latitude)],
            },
        }
        geo_data.append(geo_json_feature)
# Save geo data
    with open('./collector/static/collector/geo_data.json', 'w') as fout:
        fout.write(json.dumps(geo_data, indent=4))

    context = {'tweets_list': tweets_list}
    return render(request, 'collector/geo.html', context)


def detail(request, tweet_id):
    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    sentiment = tag_tweet_sentiment(tweet)
    return render(request, 'collector/detail.html', {'tweet': tweet, 'sentiment': sentiment})


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(
            token) else token.lower() for token in tokens]
    return tokens


def bayes_classifier(tagged_tweets, not_tagged_tweets):
    tweets_list = tagged_tweets
    global word_features
    word_features = get_word_features(get_words_in_tweets(tweets_list))
    training_set = nltk.classify.apply_features(extract_features, tweets_list)
    tweet = tweets_list[0]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    for tweet in not_tagged_tweets:
        sentiment3 = classifier.classify(extract_features(tweet.content.split()))
        tweet.tweet_sentiment = sentiment3
        tweet.save()


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


def tweet_tokenize(request, tweet_id):

    tweet = get_object_or_404(TwitterData, pk=tweet_id)
    analyze_tweet = tag_sentence(nltk.pos_tag(preprocess(tweet.content.lower())))
    sentiment = lexicon_tweet(tweet.content.lower())
    context = {'tweet': tweet, 'analyze_tweet': analyze_tweet, 'sentiment': sentiment}
    return render(request, 'collector/tokenize.html', context)


def instructions(request):
    return render(request, 'collector/instructions.html')


def tweet_for_vote(request):
    tweets_list = TwitterData.objects.all()
    n = len(tweets_list)
    t = random.randint(0, n)
    tweet = tweets_list[t]
    if len(tweet.sentiment_set.all()) == 0:
        tweet.sentiment_set.create(sentiment_text="IRR")
        tweet.sentiment_set.create(sentiment_text="UND")
        tweet.sentiment_set.create(sentiment_text="NEG")
        tweet.sentiment_set.create(sentiment_text="POS")

    return render(request, 'collector/vote.html', {'tweet': tweet})


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


def compare_file(option,event):
    return "json_data_" + option + "_" + event


def compare(request, option1, option2, event):
    file1 = open(compare_file(option1, event), "r")
    context = {}
    context1 = json.loads(file1.readline())
    for a in context1:
        context.setdefault("OP1_" + a, context1[a])
    file2 = open(compare_file(option2, event), "r")
    context2 = json.loads(file2.readline())
    for a in context2:
        context.setdefault("OP2_" + a, context2[a])
    return render(request, 'collector/compare.html',context)


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
