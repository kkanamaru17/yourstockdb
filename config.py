import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Set the session lifetime to, for example, 5 days
    PERMANENT_SESSION_LIFETIME = timedelta(days=5)
    # Optionally, you can control the "remember me" cookie lifetime
    REMEMBER_COOKIE_DURATION = timedelta(days=5)