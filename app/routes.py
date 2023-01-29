import os

from flask import render_template, redirect, request, flash, url_for, Response

from werkzeug.utils import secure_filename
from app import app
from app import detector, video

import cv2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])

model_ID = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home Page')


@app.route('/detectnow')
def detectnow():
    return render_template('detectnow.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_name)

        detector.detect_from_file(save_name)

        context = {}
        context['image_shape'] = detector.image_shape
        context['origin_image_shape'] = detector.origin_image_shape

        context['models'] = []
        for i, model_dict in enumerate(detector.models):
            context['models'].append(model_dict)

        return render_template('detectnow.html', filename=filename, context=context)
    else:
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/settings/all')
def settings():
    context = {}
    context['title'] = 'Detector parameters:'
    context['parameters'] = []
    context['parameters'].append({'name': 'CUDA', 'value': detector.device_properties, })
    for i, model_dict in enumerate(detector.models):
        context['parameters'].append({'name': f'Model {i}', 'value': model_dict['name'], 'advvalue': model_dict['score_threshold'], })
    #context['parameters'].append({'name': 'score threshold', 'value': detector.score_threshold, })
    return render_template("settings.html", context=context)


def gen(video, ID):
    while True:
        success, image = video.read()

        # perform to detect objects
        if success:
            frame = detector.show_frame(image, ID)
        else:
            img = cv2.imread(detector.path_images+'notavailable.jpg')
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            video = cv2.VideoCapture(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed/<int:ID>')
@app.route('/video_feed', defaults={'ID': 0})  # Альтернативый вариант передачи ID модели
def video_feed(ID: int):
    return Response(gen(video, ID=model_ID),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/webcam', methods=['POST', 'GET'])
def webcam():
    if request.method == 'POST':
        global model_ID
        model_ID = int(request.form.get('ModelSelect'))

    context = {}
    context['model_ID'] = model_ID
    context['models'] = []

    for i, model_dict in enumerate(detector.models):
        context['models'].append({'ID': i,
                                  'model_name': model_dict['name']})
    return render_template("webcam.html", context=context)
