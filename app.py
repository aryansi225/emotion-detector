import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
from predictor import predict, get_model

app = Flask(__name__)

get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=["POST"])
def output():
    if request.method == 'POST':
        files = os.listdir("static/uploads/")
        for file in files:
            os.remove("static/uploads/"+file)
        f = request.files['data']
        fileName = secure_filename(f.filename)
        dir = "static/uploads/" + fileName
        f.save(dir)
        predictions = predict(dir)
        pos = predictions[0].argmax() + 1
        return render_template('index.html', hasImage=True, data = dir, pos = pos, predictions = predictions)

if __name__=='__main__':
    app.run(debug=False, threaded=False)