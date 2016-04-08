import datetime
import json
from .models import TwitterData
from celery.decorators import task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def load_tweets(tweets_file):
    count = 0
    # file_path = '/static/TwitterData240320161624.txt'
    # tweets_file = open(file_path, "r")
    for line in tweets_file:
            tweet = json.loads(line)
            tweet_db = TwitterData(tweet_id=tweet['id_str'],
                                   content=tweet['text'],
                                   user=tweet['user']['screen_name'],
                                   date=datetime.datetime.strptime(
                                   str(tweet['created_at']),
                                   "%a %b %d %H:%M:%S +%f %Y"))
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
    return count


@task(name="Load_Tweets")
def load_file_task(file):
    return load_tweets(file)
