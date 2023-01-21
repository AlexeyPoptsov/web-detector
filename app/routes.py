import os
from flask import render_template, redirect, request, flash, url_for
from werkzeug.utils import secure_filename
from app import app
from app import detector

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])


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
    # flash(os.path.join(app.config['UPLOAD_FOLDER']))
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        # flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_name)
        detector.get_result(save_name)

        context = {}
        context['image_shape'] = detector.image_shape
        context['origin_image_shape'] = detector.origin_image_shape

        context['models'] = []
        for i, model_dict in enumerate(detector.models):
            context['models'].append(model_dict)

        return render_template('detectnow.html', filename=filename, context=context)
    else:
        # flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/settings/all')
def settings():
    context = {}
    context['title'] = 'Detector parameters:'
    context['parameters'] = []
    context['parameters'].append({'name': 'CUDA', 'value': detector.device_properties, })
    for i, model_dict in enumerate(detector.models):
        context['parameters'].append({'name': f'Model {i}', 'value': model_dict['name'], })
    context['parameters'].append({'name': 'score threshold', 'value': detector.score_threshold, })
    return render_template("settings.html", context=context)


@app.route('/table')
def table():
    return render_template("table.html", title='Settings page')


def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y + h, x:x + w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    # Set to global because we refer the video variable on global scope,
    # Or in other words outside the function
    global video

    # Return the result on the web
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
