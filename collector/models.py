from django.db import models

# Create your models here.
class TwitterData(models.Model):
    tweet_id= models.CharField(max_length=25) #Verificar el tipo
	content= models.CharField(max_length=200) #Verificar el tipo
	lang= models.CharField(null=True,max_length=5) #Verificar el tipo
    latitude = models.DecimalField(null=True,max_digits=11, decimal_places=7)
    longitude = models.DecimalField(null=True,max_digits=11, decimal_places=7)
    date = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=250, null=True, blank=True)
    user = models.CharField(max_length=300, null=True)
    user_location = models.CharField(max_length=300, null=True, blank=True)

    def __unicode__(self):
        return "Id: " + self.tweet_id+ "  Contet: " + self.content