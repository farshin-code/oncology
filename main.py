# flask app
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from models.mri import run
from models.ct import run as run_ct
import io
import os
import random

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predict-mri", methods=["POST"])
def predict_mri():
    if "file" not in request.files:
        return "No file"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        file_content = file.read()
        return render_template("index.html", result=run(io.BytesIO(file_content)))


@app.route("/predict-ct", methods=["POST"])
def predict_ct():
    if "file" not in request.files:
        return "No file"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        file_content = file.read()
        return render_template("index.html", result=run_ct(io.BytesIO(file_content)))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
