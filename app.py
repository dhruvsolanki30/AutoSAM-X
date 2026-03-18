from flask import Flask, render_template, request, jsonify
import os

from modules.brain.main import run_pipeline

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

        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 🔥 RUN PIPELINE
        result = run_pipeline(filepath)

        # 🔥 RETURN JSON (MATCHING YOUR JS)
        return jsonify({
            "detection": result.get("detection"),
            "confidence": result.get("confidence"),
            "tumor_area": result.get("tumor_area"),
            "severity": result.get("severity"),
            "image_url": "/" + result.get("output_image")  # VERY IMPORTANT
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