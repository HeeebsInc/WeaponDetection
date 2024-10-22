from flask import Flask, redirect, render_template, request, sessions, flash, url_for
from werkzeug.utils import secure_filename
import io
import numpy as np
import cv2
import ModelFunc as mf


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg',}
UPLOAD_FOLDER = 'templates/upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('homepage.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_picture_normal', methods=['GET', 'POST'])
def upload_picture_normal():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No image file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # file.save(secure_filename(file.filename))
            # print(file, file.filename)
            # print(file.read())
            f = file.read()
            np_img = np.fromstring(f, np.uint8)
            # print(np_img, np_img.shape)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (150,150), interpolation = cv2.INTER_CUBIC)
            # print(img, img.shape)
            temp_f = file.filename[::-1]
            temp_f = f"{temp_f[temp_f.find('.')+1:][::-1]}"
            new_path = f'NN_Weapon_Detection/FlaskApp/static/{temp_f}.jpg'
            cv2.imwrite(new_path, img)
            dim = (150,150)
            model = mf.get_conv_model()
            prob, cat, new_img, lime_img = mf.get_img_prediction_bounding_box(new_path, dim, model)
            cv2.imwrite(new_path, new_img)
            cv2.imwrite(f'NN_Weapon_Detection/FlaskApp/static/{temp_f}Lime.jpg', lime_img)
            print(prob, cat)
            if cat in ['Rifle', 'Handgun']:
                cat = 'Weapon'
            prob = f'{int(prob)}%'
            return render_template('display_photo.html', bounding_image = f'{temp_f}.jpg', lime_image = f'{temp_f}Lime.jpg',
                                   cat = cat, prob = prob)
            # return f'<img src="{{ url_for("static", filename = '{temp_f}')}}">'

    return '''
    <!doctype html>
    <title>CNN Model: Upload Any Image</title>
    <h1>CNN Model: Upload Any Image</h1>
    <p>This may take a few minutes to load...</p>
    <p>Images will be displayed in lower resolution due to resizing</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Predict>
    </form>
    <p><a href = '/'>Homepage</a>
    '''

@app.route('/upload_picture_mobilenet', methods=['GET', 'POST'])
def upload_picture_mobilenet():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No image file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # file.save(secure_filename(file.filename))
            # print(file, file.filename)
            # print(file.read())
            f = file.read()
            np_img = np.fromstring(f, np.uint8)
            # print(np_img, np_img.shape)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
            # print(img, img.shape)
            temp_f = file.filename[::-1]
            temp_f = f"{temp_f[temp_f.find('.')+1:][::-1]}"
            new_path = f'NN_Weapon_Detection/FlaskApp/static/{temp_f}.jpg'
            cv2.imwrite(new_path, img)
            dim = (224,224)
            model = mf.get_mobilenet()
            prob, cat, new_img, lime_img = mf.get_img_prediction_bounding_box(new_path, dim, model)
            cv2.imwrite(new_path, new_img)
            cv2.imwrite(f'NN_Weapon_Detection/FlaskApp/static/{temp_f}Lime.jpg', lime_img)
            print(prob, cat)
            if cat in ['Rifle', 'Handgun']:
                cat = 'Weapon'
            prob = f'{int(prob)}%'
            return render_template('display_photo.html', bounding_image = f'{temp_f}.jpg', lime_image = f'{temp_f}Lime.jpg',
                                   cat = cat, prob = prob)
            # return f'<img src="{{ url_for("static", filename = '{temp_f}')}}">'

    return '''
    <!doctype html>
    <title>Mobilenet: Upload Any Image</title>
    <h1>Mobilenet: Upload Any Image</h1>
    <p>This may take a few minutes to load...</p>
    <p>Images will be displayed in lower resolution due to resizing</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Predict>
    </form>
    <p><a href = '/'>Homepage</a>
    '''
if __name__ == '__main__':
    app.run(threaded = True, port = 5000)