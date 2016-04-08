from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField()


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
