from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model import generate_caption

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            caption, confidence = generate_caption(image_path)
    return render_template('index.html', caption=caption, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
