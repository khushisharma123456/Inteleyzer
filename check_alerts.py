import os
os.environ['SKIP_AUTO_POPULATE'] = '1'
from flask import Flask
from models import db, Alert

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "inteleyzer.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    alerts = Alert.query.limit(10).all()
    print('Sample alerts:')
    for a in alerts:
        print(f'ID: {a.id}, Drug: {a.drug_name}, RecipientType: {a.recipient_type}, Sender: {a.sender.name if a.sender else "None"}')
    print(f'\nTotal alerts: {Alert.query.count()}')
