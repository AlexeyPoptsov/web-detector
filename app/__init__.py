from flask import Flask

from app.detector import Detector
from config import Config
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

from flask_bootstrap import Bootstrap

import cv2

# from flask_socketio import SocketIO
# from OpenSSL import SSL
#
# context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# context.use_privatekey_file('server.key')
# context.use_certificate_file('server.crt')

app = Flask(__name__,
            static_folder='/path/to/static',
            template_folder='/path/to/templates')

app = Flask(__name__)
app.config.from_object(Config)

# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

bootstrap = Bootstrap(app)
detector = Detector(model_IDs = [0,1,2,3,4,5], path_images=app.config['UPLOAD_FOLDER'])

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#socketio = SocketIO(app,cors_allowed_origins='*' )
# face_cascade = cv2.CascadeClassifier()

# Load the pretrained model
# face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))

#from app import routes, models

from app import routes