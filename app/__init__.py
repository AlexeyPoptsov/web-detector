from flask import Flask

from app.detector import Detector
from config import Config

from flask_bootstrap import Bootstrap

import cv2

app = Flask(__name__,
            static_folder='/path/to/static',
            template_folder='/path/to/templates')

app = Flask(__name__)
app.config.from_object(Config)

bootstrap = Bootstrap(app)
detector = Detector(model_IDs=[0, 1, 2, 3, 4, 5, 10], path_images=app.config['UPLOAD_FOLDER'])

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

from app import routes
