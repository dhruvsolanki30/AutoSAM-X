from flask import Flask, request, render_template, send_from_directory
import os

from modules.kidney.main import run_pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(RESULT_FOLDER,exist_ok=True)


@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/upload",methods=["POST"])
def upload():

    file = request.files["file"]
    option = request.form["option"]

    path = os.path.join(UPLOAD_FOLDER,file.filename)

    file.save(path)

    run_pipeline(path,option)

    return render_template("homepage.html",done=True)


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory("results",filename)


if __name__ == "__main__":
    app.run(debug=True)