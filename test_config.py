from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

print("Database URI:", app.config.get('SQLALCHEMY_DATABASE_URI'))
