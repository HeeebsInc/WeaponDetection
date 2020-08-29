from flask import Flask, redirect, render_template, request, sessions, flash, url_for
from werkzeug.utils import secure_filename
import io
import numpy as np
import cv2
from ModelFunc import get_img_prediction_bounding_box


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

@app.route('/upload_picture', methods=['GET', 'POST'])
def upload_picture():
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
            img = cv2.resize(img, (96,96), interpolation = cv2.INTER_CUBIC)
            # print(img, img.shape)
            temp_f = file.filename[::-1]
            temp_f = f"{temp_f[temp_f.find('.')+1:][::-1]}"
            new_path = f'NN_Weapon_Detection/FlaskApp/static/{temp_f}.jpg'
            cv2.imwrite(new_path, img)
            prob, cat, new_img, lime_img = get_img_prediction_bounding_box(new_path)
            cv2.imwrite(new_path, new_img)
            cv2.imwrite(f'NN_Weapon_Detection/FlaskApp/static/{temp_f}Lime.jpg', lime_img)
            print(prob, cat)
            if cat in ['Rifle', 'Handgun']:
                cat = 'Weapon'
            prob = f'{int(prob*100)}%'
            return render_template('display_photo.html', bounding_image = f'{temp_f}.jpg', lime_image = f'{temp_f}Lime.jpg',
                                   cat = cat, prob = prob)
            # return f'<img src="{{ url_for("static", filename = '{temp_f}')}}">'

    return '''
    <!doctype html>
    <title>Upload Any Image</title>
    <h1>Upload new File</h1>
    <p>This may take a few minutes...</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

# @app.route('/upload_picture')
# def upload_picture():
#     return render_
if __name__ == '__main__':
    app.run(threaded = True, port = 5000)