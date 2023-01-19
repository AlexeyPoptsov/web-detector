import os
from flask import render_template, redirect, request, flash, url_for
from werkzeug.utils import secure_filename
from app import app


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home Page')


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detectnow')
def upload_form():
	return render_template('/detectnow.html')

@app.route('/detectnow', methods=['POST'])
def detectnow():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # else:
    #	flash('Allowed image types are -> png, jpg, jpeg, gif')
    #	return redirect(request.url)
    return render_template('detectnow.html', filenames=file_names)


@app.route('/settings')
def settings():
    return render_template("settings.html", title='Settings page')

@app.route('/table')
def table():
    return render_template("table.html", title='Settings page')



@app.route('/display_image/<filename>')
def display_image(filename):
    flash('full file name : ' + os.path.join(app.config['UPLOAD_FOLDER']) + filename)
    return redirect(url_for('', filename=+os.path.join(app.config['UPLOAD_FOLDER']) + filename), code=301)
