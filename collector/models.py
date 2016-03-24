from django.db import models

# Create your models here.
class TwitterData(models.Model):
	tweet_id= models.CharField(max_length=25) 
	content= models.CharField(max_length=200) 
	user = models.CharField(max_length=300, null=True)
	date = models.DateTimeField(null=True, blank=True)

	latitude = models.DecimalField(null=True,max_digits=11, decimal_places=7)
    longitude = models.DecimalField(null=True, max_digits=11, decimal_places=7)
	source = models.CharField(max_length=250, null=True, blank=True)
	user_location = models.CharField(max_length=300, null=True, blank=True)
	lang= models.CharField(null=True,max_length=5) 
	
  
    def __unicode__(self):
        return "id: " + self.tweet_id + "  content " + self.content