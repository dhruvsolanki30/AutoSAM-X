from flask import Flask, render_template, request, jsonify
import os

from modules.brain.main import run_pipeline

from modules.brain.hemorrhage import run_hemorrhage
from modules.brain.stroke import run_stroke

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/brain")
def brain():
    return render_template("brain.html")


# 🔥 MAIN API FOR YOUR UI (IMPORTANT)
@app.route("/predict", methods=["POST"])
def predict():

    try:
        file = request.files.get("file")
        pathology = request.form.get("pathology", "").lower()

        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 🔥 ROUTING BASED ON PATHOLOGY
        if "tumor" in pathology:
            result = run_pipeline(filepath)

        elif "hemorrhage" in pathology:
            result = run_hemorrhage(filepath)

        elif "stroke" in pathology:
            result = run_stroke(filepath)

        else:
            return jsonify({"error": "Unsupported pathology"}), 400

        return jsonify({
            "detection": result.get("detection"),
            "confidence": result.get("confidence"),
            "tumor_area": result.get("tumor_area", 0),
            "severity": result.get("severity"),
            "image_url": result.get("output_image")
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# Other organs (disabled)
@app.route("/liver")
def liver():
    return "<h1>Coming Soon</h1>"


@app.route("/kidney")
def kidney():
    return "<h1>Coming Soon</h1>"


@app.route("/lungs")
def lungs():
    return "<h1>Coming Soon</h1>"


if __name__ == "__main__":
    app.run(debug=True)