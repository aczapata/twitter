from django import forms
from collector.models import TweetFile
from django.forms import ModelForm


class UploadFileForm(ModelForm):
    class Meta:
            model = TweetFile
            fields = ['file']


class FilterForm(forms.Form):
    #Parties
    dp = forms.BooleanField(required=False)
    rp = forms.BooleanField(required=False)

    #Candidates
    dt = forms.BooleanField(required=False)
    hc = forms.BooleanField(required=False)
    mr = forms.BooleanField(required=False)
    bs = forms.BooleanField(required=False)
    tc = forms.BooleanField(required=False)

    #Events
    st = forms.BooleanField(required=False)
    fp = forms.BooleanField(required=False)
    rd = forms.BooleanField(required=False)
    dd = forms.BooleanField(required=False)

class SearchForm(forms.Form):
    text = forms.CharField()
"""
    CHOICES = (
        ('HC', 'Hillary Clinton'),
        ('DT', 'Donald Trump'),
        ('BS', 'Bernie Sanders'),
        ('MR', 'Marco Rubio'),
    )
	#stream.filter(track=["Hillary Clinton", "Donald Trump", "Bernie Sanders", "Marco Rubio", "US election 2016", "Ted Cruz", "John Kasich", "Ben Carson", "Republicans", "Democrats", "Clinton", "Trump", "Bernie", "Rubio", "Cruz", "Carson", "Kasich", "US primaries", "uselections", "Election2016"])
    text= forms.ChoiceField(choices= CHOICES)
"""
