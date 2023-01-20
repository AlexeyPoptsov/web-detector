from flask import Flask

from app.detector import Detector
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from flask_bootstrap import Bootstrap

# app = Flask(__name__,
#             static_folder='/path/to/static',
#             template_folder='/path/to/templates')

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
bootstrap = Bootstrap(app)
detector = Detector(model_IDs = [0,1,2], path_images=app.config['UPLOAD_FOLDER'])

from app import routes, models