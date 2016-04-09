"""
WSGI config for twitter project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.8/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from whitenoise.django import DjangoWhiteNoise
import djcelery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "twitter.settings")

application = get_wsgi_application()
application = get_wsgi_application()
application = DjangoWhiteNoise(application)
djcelery.setup_loader()
