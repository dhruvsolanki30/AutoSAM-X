from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import traceback

# Brain pathologies
try:
    from modules.brain.main import run_pipeline as run_brain_tumor
    from modules.brain.hemorrhage import run_hemorrhage
    from modules.brain.stroke import run_stroke
    BRAIN_AVAILABLE = True
except Exception as e:
    print(f"Warning: Brain modules not available: {e}")
    BRAIN_AVAILABLE = False

# Lung pathologies
try:
    from modules.lung.analyzer import PathologyAnalyzer
    LUNG_AVAILABLE = True
    lung_analyzer = PathologyAnalyzer()
except Exception as e:
    print(f"Warning: Lung modules not available: {e}")
    LUNG_AVAILABLE = False

# Kidney pathologies
try:
    from modules.kidney.main import run_pipeline as run_kidney_analysis
    KIDNEY_AVAILABLE = True
except Exception as e:
    print(f"Warning: Kidney modules not available: {e}")
    KIDNEY_AVAILABLE = False

# Liver pathologies
try:
    from modules.liver.main import run_pipeline as run_liver_analysis
    LIVER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Liver modules not available: {e}")
    LIVER_AVAILABLE = False

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ========== HOME & ORGAN ROUTES ==========
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/brain")
def brain():
    if BRAIN_AVAILABLE:
        return render_template("brain.html")
    return "<h1>Brain module not available</h1>", 503


@app.route("/lungs")
def lungs():
    if LUNG_AVAILABLE:
        return render_template("lungs.html")
    return "<h1>Lung module not available</h1>", 503


@app.route("/kidney")
def kidney():
    if KIDNEY_AVAILABLE:
        return render_template("kidney.html")
    return "<h1>Kidney module not available</h1>", 503


@app.route("/liver")
def liver():
    if LIVER_AVAILABLE:
        return render_template("liver.html")
    return "<h1>Liver module not available</h1>", 503


# ========== UNIFIED PREDICTION API ==========
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        organ = request.form.get("organ", "").lower()
        pathology = request.form.get("pathology", "").lower()

        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = {}

        # ===== BRAIN PATHOLOGIES =====
        if organ == "brain":
            if not BRAIN_AVAILABLE:
                return jsonify({"error": "Brain module not available"}), 503
            
            if "tumor" in pathology:
                result = run_brain_tumor(filepath)
            elif "hemorrhage" in pathology:
                result = run_hemorrhage(filepath)
            elif "stroke" in pathology:
                result = run_stroke(filepath)
            else:
                return jsonify({"error": "Unsupported brain pathology"}), 400

        # ===== LUNG PATHOLOGIES =====
        elif organ == "lungs" or organ == "lung":
            if not LUNG_AVAILABLE:
                return jsonify({"error": "Lung module not available"}), 503
            
            # Use lung analyzer for multi-pathology analysis
            try:
                result = lung_analyzer.analyze(filepath, pathology.lower())
                if isinstance(result, dict):
                    result = {
                        "detection": result.get("detection", "Detected"),
                        "confidence": result.get("confidence", 0.95),
                        "severity": result.get("severity", "Medium"),
                        "image_url": result.get("image_url", ""),
                        "findings": result.get("findings", {})
                    }
            except Exception as e:
                return jsonify({"error": f"Lung analysis failed: {str(e)}"}), 500

        # ===== KIDNEY PATHOLOGIES =====
        elif organ == "kidney":
            if not KIDNEY_AVAILABLE:
                return jsonify({"error": "Kidney module not available"}), 503
            
            result = run_kidney_analysis(filepath)

        # ===== LIVER PATHOLOGIES =====
        elif organ == "liver":
            if not LIVER_AVAILABLE:
                return jsonify({"error": "Liver module not available"}), 503
            
            result = run_liver_analysis()

        else:
            return jsonify({"error": f"Unsupported organ: {organ}"}), 400

        # Standardize response format
        return jsonify({
            "detection": result.get("detection", "Detected"),
            "confidence": result.get("confidence", 0.9),
            "severity": result.get("severity", "Unknown"),
            "tumor_area": result.get("tumor_area", 0),
            "image_url": result.get("image_url", result.get("output_image", "")),
            "findings": result.get("findings", {})
        })

    except Exception as e:
        print(f"ERROR in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# ========== STATUS ENDPOINT ==========
@app.route("/status")
def status():
    return jsonify({
        "brain_available": BRAIN_AVAILABLE,
        "lung_available": LUNG_AVAILABLE,
        "kidney_available": KIDNEY_AVAILABLE,
        "liver_available": LIVER_AVAILABLE
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)