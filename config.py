import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = 'postgresql://stockdb_postgresql_user:fWoDaXdOFpQ37GGOCG5CyYiQTxxEc7Fb@dpg-crdsu33v2p9s73cmnlf0-a.oregon-postgres.render.com/stockdb_postgresql'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Set the session lifetime to, for example, 5 days
    PERMANENT_SESSION_LIFETIME = timedelta(days=5)
    # Optionally, you can control the "remember me" cookie lifetime
    REMEMBER_COOKIE_DURATION = timedelta(days=5)