# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-04-30 23:51
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('collector', '0007_auto_20160417_1826'),
    ]

    operations = [
        migrations.AddField(
            model_name='twitterdata',
            name='tweet_sentiment',
            field=models.CharField(blank=True, max_length=15, null=True),
        ),
    ]
