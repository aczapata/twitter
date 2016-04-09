from django.db import models

# Create your models here.


class TwitterData(models.Model):

    tweet_id = models.CharField(unique=True, max_length=25)
    content = models.CharField(max_length=200)
    lang = models.CharField(null=True, max_length=5, blank=True)
    latitude = models.DecimalField(null=True, max_digits=11,
                                   decimal_places=7, blank=True)
    longitude = models.DecimalField(null=True, max_digits=11,
                                    decimal_places=7, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=250, null=True, blank=True)
    user = models.CharField(max_length=300, null=True)
    user_location = models.CharField(max_length=300, null=True, blank=True)

    def __unicode__(self):
        return "Id: " + self.tweet_id + "  Content: " + self.content


class Sentiment(models.Model):

    POSITIVE = 'POS'
    NEGATIVE = 'NEG'
    UNDECIDABLE = 'UND'
    IRRELEVANT = 'IRR'
    VOTE_CHOICES = (
        (POSITIVE, 'Positive'),
        (NEGATIVE, 'Negative'),
        (UNDECIDABLE, 'Undecidable'),
        (IRRELEVANT, 'Irrelevant'),
    )
    sentiment_text = models.CharField(
        max_length=3, choices=VOTE_CHOICES, default=UNDECIDABLE)
    votes = models.IntegerField(default=0)
    tweet = models.ForeignKey(TwitterData, on_delete=models.CASCADE)

    def __unicode__(self):
        return "Id: " + self.tweet.tweet_id + "  Type: " + self.sentiment_text


class TweetFile(models.Model):
    file = models.FileField()
