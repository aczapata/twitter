from django.db import models

# Create your models here.
class TwitterData(models.Model):
    latitude = models.DecimalField(max_digits=11, decimal_places=7)
    longitude = models.DecimalField(max_digits=11, decimal_places=7)
    date = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=250, null=True, blank=True)
    user = models.CharField(max_length=300, null=True)
    user_location = models.CharField(max_length=300, null=True, blank=True)

    def __unicode__(self):
        return "User: " + self.user + "  LatLon: " + str(self.latitude) + ", " + str(self.longitude)