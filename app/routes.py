import os
import numpy as np
from flask import render_template, redirect, request, flash, url_for, Response

from werkzeug.utils import secure_filename
from app import app
from app import detector, video

import cv2
import time

from flask_socketio import SocketIO, emit
from PIL import Image
import base64
import numpy as np
import pyshine as ps

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
model_ID = 0
FPS = np.zeros(5)
FPS_index = 0



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


def gen(video, ID):
    while True:
        start_time = time.time()
        success, image = video.read()

        # perform the resizing
        # h = 320.0
        # r = h / image.shape[0]
        # dim = (int(image.shape[1] * r), int(h))
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # perform to detect objects
        image, count_classes = detector.detect(image, ID)

        # draw classes block
        x = image.shape[1] - 200
        y = image.shape[0] - 150
        cv2.putText(image, 'Detected objects', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192), 2)
        image = cv2.rectangle(image, (x, y), (x + 200 -2 , y + 200-2), (192, 192, 192), 1)
        step = 20
        for _class in count_classes:
            cv2.putText(image,
                        f'{_class}: {count_classes[_class]}',
                        (x + 10, y + step),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192),1)
            step +=20

        work_time = time.time() - start_time

        global FPS, FPS_index
        FPS_index = (FPS_index + 1) % 5
        FPS[FPS_index] = (1 / work_time)
        cv2.putText(image, f'({image.shape[1]}x{image.shape[0]}), FPS: {FPS.mean():.2f}',
                    (x+10, y+140), cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192), 1)


        ret, jpeg = cv2.imencode('.jpg',image)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed/<int:ID>')
@app.route('/video_feed', defaults={'ID': 0})  # added line which handles None
def video_feed(ID: int):
    # Set to global because we refer the video variable on global scope,
    # Or in other words outside the function
    # global video

    # Return the result on the web
    return Response(gen(video, ID=model_ID),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_client')
def video_client():
    return render_template("video_client.html")


@app.route('/webcam', methods=['POST','GET'])
def webcam():
    if request.method == 'POST':
        global model_ID
        # if request.form.get('ModelSelect') =='0':
        #     model_ID = 0
        # elif request.form.get('ModelSelect') =='1':
        #     model_ID = 1
        # elif request.form.get('ModelSelect') =='2':
        #     model_ID =2
        # else:
        model_ID = int(request.form.get('ModelSelect'))

    context = {}
    context['model_ID'] = model_ID
    context['models'] = []

    for i, model_dict in enumerate(detector.models):
        context['models'].append({'ID': i,
                                  'model_name': model_dict['name']})
    return render_template("webcam.html", context=context)




# def readb64(base64_string):
#     idx = base64_string.find('base64,')
#     base64_string = base64_string[idx + 7:]
#
#     sbuf = io.BytesIO()
#
#     sbuf.write(base64.b64decode(base64_string, ' /'))
#     pimg = Image.open(sbuf)
#
#     return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
#
#
# def moving_average(x):
#     return np.mean(x)
#
#
# def getMaskOfLips(img, points):
#     """ This function will input the lips points and the image
#         It will return the mask of lips region containing white pixels
#     """
#     mask = np.zeros_like(img)
#     mask = cv2.fillPoly(mask, [points], (255, 255, 255))
#     return mask
#
#
# def changeLipstick(img, value):
#     """ This funciton will take img image and lipstick color RGB
#         Out the image with a changed lip color of the image
#     """
#
#     img = cv2.resize(img, (0, 0), None, 1, 1)
#     imgOriginal = img.copy()
#     imgColorLips = imgOriginal
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(imgGray)
#
#     for face in faces:
#         x1, y1 = face.left(), face.top()
#         x2, y2 = face.right(), face.bottom()
#
#         facial_landmarks = predictor(imgGray, face)
#         points = []
#         for i in range(68):
#             x = facial_landmarks.part(i).x
#             y = facial_landmarks.part(i).y
#             points.append([x, y])
#
#         points = np.array(points)
#         imgLips = getMaskOfLips(img, points[48:61])
#
#         imgColorLips = np.zeros_like(imgLips)
#
#         imgColorLips[:] = value[2], value[1], value[0]
#         imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
#
#         value = 1
#         value = value // 10
#         if value % 2 == 0:
#             value += 1
#         kernel_size = (6 + value, 6 + value)  # +1 is to avoid 0
#
#         weight = 1
#         weight = 0.4 + (weight) / 400
#         imgColorLips = cv2.GaussianBlur(imgColorLips, kernel_size, 10)
#         imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, weight, 0)
#
#     return imgColorLips
#
#
# @socketio.on('catch-frame')
# def catch_frame(data):
#     emit('response_back', data)
#
#
# global fps, prev_recv_time, cnt, fps_array
# fps = 30
# prev_recv_time = 0
# cnt = 0
# fps_array = [0]
#
#
# @socketio.on('image')
# def image(data_image):
#     global fps, cnt, prev_recv_time, fps_array
#     recv_time = time.time()
#     text = 'FPS: ' + str(fps)
#     frame = (readb64(data_image))
#
#     frame = changeLipstick(frame, [255, 0, 0])
#     frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0,
#                         background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
#     imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
#
#     # base64 encode
#     stringData = base64.b64encode(imgencode).decode('utf-8')
#     b64_src = 'data:image/jpeg;base64,'
#     stringData = b64_src + stringData
#
#     # emit the frame back
#     emit('response_back', stringData)
#
#     fps = 1 / (recv_time - prev_recv_time)
#     fps_array.append(fps)
#     fps = round(moving_average(np.array(fps_array)), 1)
#     prev_recv_time = recv_time
#     # print(fps_array)
#     cnt += 1
#     if cnt == 30:
#         fps_array = [fps]
#         cnt = 0
