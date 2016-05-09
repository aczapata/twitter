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


class CompareForm(forms.Form):
    #Parties
    options = ["Choice1", "Choice2", "Choice 3"]
    choices_candidates = (
        ('RP', 'Republican Party'),
        ('DP', 'Democratic Party'),
        ('DT', 'Donald Trump'),
        ('HC', 'Hillary Clinton'),
        ('BS', 'Bernie Sanders'),
        ('MR', 'Marco Rubio'),
        ('TC', 'Ted Cruz'),
    )
    choices_dates = (
        ('AA', 'All'),
        ('ST', 'Super Thursday'),
        ('FP', 'Florida Primaries'),
        ('RD', 'Republican Debate'),
        ('DD', 'Democratic Debate'),
    )
    option1 = forms.ChoiceField(choices=choices_candidates)
    option2 = forms.ChoiceField(choices=choices_candidates)
    event = forms.ChoiceField(choices=choices_dates)

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
