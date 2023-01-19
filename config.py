import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mssql+pymssql://sql_admin:Ap30875@192.168.2.131:1433/web-detector'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # path = os.getcwd()
    # # file Upload
    # UPLOAD_FOLDER = os.path.join(path, 'uploads')

    UPLOAD_FOLDER = 'app/static/uploads/'
    MAX_CONTENT_LENGTH = 3 * 3000 * 1500
