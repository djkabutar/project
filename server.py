from werkzeug.utils import secure_filename, redirect
from flask import Flask, request, url_for, render_template
import numpy as np
from PIL import Image
from ISR.models import RRDN
from io import BytesIO
import base64

app = Flask(__name__)

IMAGE_FOLDER = "Image/"
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

def processed_image(img):
    img = Image.open(img)
    lr_img = np.array(img)

    rdn = RRDN(weights='gans')
    sr_img = rdn.predict(lr_img, by_patch_of_size=5)
    Image.fromarray(sr_img)

    response = im_2_b64(Image.fromarray(sr_img))

    return response

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(app.config['UPLOAD_FOLDER'] + secure_filename(file.filename))
        return processed_image(app.config['UPLOAD_FOLDER'] + secure_filename(file.filename))
        # return "File Uploaded Successfully"
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
