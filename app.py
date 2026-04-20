from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import traceback
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Brain pathologies
try:
    from brain.main import run_pipeline as run_brain_tumor
    from brain.hemorrhage import run_hemorrhage
    from brain.stroke import run_stroke
    BRAIN_AVAILABLE = True
except Exception as e:
    print(f"Warning: Brain modules not available: {e}")
    BRAIN_AVAILABLE = False

# Lung pathologies
try:
    from lung.analyzer import PathologyAnalyzer
    LUNG_AVAILABLE = True
    lung_analyzer = PathologyAnalyzer()
except Exception as e:
    print(f"Warning: Lung modules not available: {e}")
    LUNG_AVAILABLE = False

# Kidney pathologies
try:
    from kidney.main import run_pipeline as run_kidney_analysis
    KIDNEY_AVAILABLE = True
except Exception as e:
    print(f"Warning: Kidney modules not available: {e}")
    KIDNEY_AVAILABLE = False

# Liver pathologies
try:
    from liver.main import run_pipeline as run_liver_analysis
    LIVER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Liver modules not available: {e}")
    LIVER_AVAILABLE = False

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
LUNA16_PATH = "lung/luna16/subset0"
ANNOTATIONS_PATH = "lung/annotations.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ========== FALLBACK ANALYSIS ==========
def fallback_analysis(filepath, organ, pathology):
    """Image analysis fallback when ML models are unavailable"""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"detection": "Could not load image", "confidence": 0,
                "severity": "Unknown", "tumor_area": 0, "image_url": "", "findings": {}}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    mean_val = np.mean(enhanced)
    std_val = np.std(enhanced)
    threshold = int(min(mean_val + 1.5 * std_val, 250))
    _, bright_mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > 100]
    total_area = sum(cv2.contourArea(c) for c in significant)

    if total_area > 3000:
        severity, detection = "High", "Abnormality Detected"
    elif total_area > 1000:
        severity, detection = "Medium", "Possible Abnormality"
    elif total_area > 200:
        severity, detection = "Low", "Minor Finding"
    else:
        severity, detection = "Normal", "No Significant Finding"

    confidence = round(min(0.95, 0.3 + total_area / 10000), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Scan')
    axes[0].axis('off')

    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Enhanced')
    axes[1].axis('off')

    overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for c in significant:
        cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2)
    if significant:
        largest = max(significant, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Analysis')
    axes[2].axis('off')

    label = pathology.replace('detection', '').strip().title() or 'General'
    plt.suptitle(f"{organ.title()} - {label} Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_filename = f"result_{int(time.time())}.png"
    save_path = os.path.join(OUTPUT_FOLDER, output_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    return {
        "detection": detection,
        "confidence": confidence,
        "severity": severity,
        "tumor_area": int(total_area),
        "image_url": f"/static/outputs/{output_filename}",
        "findings": {}
    }


# ========== HOME & ORGAN ROUTES ==========
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/brain")
def brain():
    return render_template("brain.html")


@app.route("/lungs")
def lungs():
    return render_template("lungs.html")


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


@app.route("/liver")
def liver():
    return render_template("liver.html")


# ========== LUNG PIPELINE HELPER ==========

# ========== LIVER PIPELINE HELPER ==========
def _run_liver_pipeline(filepath, pathology):
    """Run liver analysis using YOLO + SAM (with intensity fallback) and 3-panel output."""
    from liver.preprocess import preprocess_slice
    from liver.detect import detect_objects
    from shared.segmentation import segment_with_bbox
    from shared.mask_utils import refine_mask, compute_mask_metrics
    import nibabel as nib

    is_nifti = filepath.endswith((".nii", ".nii.gz"))

    # ---- Load and pick best slice ----
    if is_nifti:
        nii = nib.load(filepath)
        volume = nii.get_fdata()
        start = int(volume.shape[2] * 0.3)
        end = int(volume.shape[2] * 0.8)

        # Try YOLO on ~5 evenly spaced sample slices (fast)
        sample_indices = np.linspace(start, end - 1, 5, dtype=int)
        best_slice = None
        best_det = None
        best_conf = 0.0
        yolo_worked = False

        for i in sample_indices:
            sl = volume[:, :, i].astype(np.float32)
            proc = preprocess_slice(sl)
            liver_d, tumor_d, t_found = detect_objects(proc, conf_thresh=0.15)
            if t_found and len(tumor_d) > 0:
                if tumor_d[0][1] > best_conf:
                    best_conf = tumor_d[0][1]
                    best_slice = proc
                    best_det = ("tumor", liver_d, tumor_d)
                    yolo_worked = True
            elif len(liver_d) > 0 and not yolo_worked:
                if liver_d[0][1] > best_conf:
                    best_conf = liver_d[0][1]
                    best_slice = proc
                    best_det = ("liver", liver_d, tumor_d)
                    yolo_worked = True

        # If YOLO found nothing, pick highest-variance slice
        if best_slice is None:
            stds = [volume[:, :, i].std() for i in range(start, end, 3)]
            best_idx = start + np.argmax(stds) * 3
            sl = volume[:, :, best_idx].astype(np.float32)
            best_slice = preprocess_slice(sl)

        processed = best_slice
    else:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image")
        processed = preprocess_slice(img.astype(np.float32))
        # Try YOLO on single image
        liver_d, tumor_d, t_found = detect_objects(processed, conf_thresh=0.15)
        yolo_worked = len(liver_d) > 0 or t_found
        if yolo_worked:
            best_det = ("tumor" if t_found else "liver", liver_d, tumor_d)
        else:
            best_det = None

    # ---- Detection + Segmentation ----
    h, w = processed.shape
    resized = cv2.resize(processed, (512, 512))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    scale_x, scale_y = 512 / w, 512 / h

    if yolo_worked and best_det is not None:
        # Use YOLO detections
        det_type, liver_det, tumor_det = best_det
        det_box = None
        confidence = 0.0

        if len(tumor_det) > 0:
            _, confidence, det_box = tumor_det[0]
        elif len(liver_det) > 0:
            _, confidence, det_box = liver_det[0]

        # Scale box to 512x512 for SAM
        if det_box is not None:
            bx1 = int(det_box[0] * scale_x)
            by1 = int(det_box[1] * scale_y)
            bx2 = int(det_box[2] * scale_x)
            by2 = int(det_box[3] * scale_y)
            bbox_512 = [bx1, by1, bx2, by2]
        else:
            bbox_512 = [128, 128, 384, 384]

        tumor_found = len(tumor_det) > 0
        # Scale det_box for display on 512 image
        display_box = bbox_512
    else:
        # Intensity-based fallback detection
        print("[LIVER] YOLO found nothing, using intensity-based detection")
        blurred = cv2.GaussianBlur(resized, (15, 15), 0)
        mean_val = np.mean(blurred)
        std_val = np.std(blurred)
        thresh_val = int(min(mean_val + 1.0 * std_val, 240))
        _, bright_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest)
            pad = 20
            bbox_512 = [max(0, rx - pad), max(0, ry - pad),
                        min(511, rx + rw + pad), min(511, ry + rh + pad)]
        else:
            bbox_512 = [100, 100, 412, 412]

        display_box = bbox_512
        confidence = 0.75
        tumor_found = True  # intensity found something

    # SAM segmentation using shared model
    mask = segment_with_bbox(rgb_img, bbox_512)
    refined = refine_mask(mask)
    metrics = compute_mask_metrics(refined)

    tumor_area = metrics.get("area", 0)
    detection = "Tumor Detected" if tumor_found and tumor_area > 0 else "No Tumor Detected"

    if tumor_area > 3000:
        severity = "High"
    elif tumor_area > 1000:
        severity = "Medium"
    elif tumor_area > 100:
        severity = "Low"
    else:
        severity = "Normal"

    findings = {
        "Area (pixels)": str(tumor_area),
        "Circularity": str(metrics.get("circularity", 0)),
        "Severity": severity,
    }

    print(f"[LIVER] Detection: {detection}, Area: {tumor_area}, Severity: {severity}")

    # ---- 3-panel image ----
    output_filename = f"liver_{pathology}_{int(time.time())}.png"
    save_path = os.path.join(OUTPUT_FOLDER, output_filename)

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original", color="white")
    plt.imshow(resized, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Detection", color="white")
    plt.imshow(resized, cmap="gray")
    if display_box is not None:
        x1, y1, x2, y2 = display_box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           edgecolor="red", linewidth=2, fill=False)
        )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmentation", color="white")
    plt.imshow(resized, cmap="gray")
    if np.sum(refined) > 0:
        plt.imshow(refined, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    fig.patch.set_facecolor("black")
    plt.savefig(save_path, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return {
        "detection": detection,
        "confidence": round(float(confidence), 2),
        "severity": severity,
        "tumor_area": tumor_area,
        "image_url": f"/static/outputs/{output_filename}",
        "findings": findings,
    }


# ========== LUNG PIPELINE HELPER ==========
def _run_lung_pipeline(scan_path, pathology):
    """Run lung analysis on an uploaded .mhd file, creating a synthetic annotation if needed."""
    import SimpleITK as sitk

    # Normalize pathology name (e.g., "nodule detection" -> "nodule")
    pathology = pathology.replace(" detection", "").strip()

    # Check if this scan matches a known LUNA16 annotation
    annotations = None
    annotation = None
    if os.path.exists(ANNOTATIONS_PATH):
        annotations = pd.read_csv(ANNOTATIONS_PATH)
        seriesuid = os.path.splitext(os.path.basename(scan_path))[0]
        matched = annotations[annotations.seriesuid == seriesuid]
        if not matched.empty:
            annotation = matched.iloc[0]

    if annotation is None:
        # Build a synthetic annotation by finding an interesting slice
        vol = sitk.ReadImage(scan_path)
        arr = sitk.GetArrayFromImage(vol)  # shape: (Z, Y, X)
        # Use slice with highest standard-deviation as the most interesting
        stds = np.array([arr[z].std() for z in range(arr.shape[0])])
        best_z = int(np.argmax(stds))
        cy, cx = arr.shape[1] // 2, arr.shape[2] // 2

        if pathology == "nodule":
            # Need coordX, coordY, coordZ in world coordinates
            origin = np.array(vol.GetOrigin())       # (X, Y, Z)
            spacing = np.array(vol.GetSpacing())      # (X, Y, Z)
            world_x = origin[0] + cx * spacing[0]
            world_y = origin[1] + cy * spacing[1]
            world_z = origin[2] + best_z * spacing[2]
            annotation = pd.Series({
                "coordX": world_x,
                "coordY": world_y,
                "coordZ": world_z,
                "diameter_mm": 10.0
            })
        else:
            # pneumonia / fibrosis use slice_index, center_x, center_y
            annotation = pd.Series({
                "slice_index": best_z,
                "center_x": cx,
                "center_y": cy
            })

    results = lung_analyzer.analyze(
        image_path=scan_path,
        pathology=pathology,
        annotation=annotation,
        visualize=False
    )

    # Create 3-panel image like brain (Original, Detection, Segmentation)
    output_filename = f"lung_{pathology}_{int(time.time())}.png"
    save_path = os.path.join(OUTPUT_FOLDER, output_filename)

    prepared_img = results.get("prepared_img")
    bbox = results.get("bbox")
    refined_mask = results.get("mask")

    fig = plt.figure(figsize=(12, 4))

    # Panel 1: Original
    plt.subplot(1, 3, 1)
    plt.title("Original", color="white")
    plt.imshow(prepared_img, cmap="gray")
    plt.axis("off")

    # Panel 2: Detection (with bounding box)
    plt.subplot(1, 3, 2)
    plt.title("Detection", color="white")
    plt.imshow(prepared_img, cmap="gray")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                           edgecolor="red", linewidth=2, fill=False)
        )
    plt.axis("off")

    # Panel 3: Segmentation (with mask overlay)
    plt.subplot(1, 3, 3)
    plt.title("Segmentation", color="white")
    plt.imshow(prepared_img, cmap="gray")
    if refined_mask is not None and np.sum(refined_mask) > 0:
        plt.imshow(refined_mask, cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    fig.patch.set_facecolor("black")
    plt.savefig(save_path, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    findings = {k: str(v) for k, v in results.get("findings", {}).items()}
    return {
        "detection": f"{results['pathology_name']} Detected",
        "confidence": 0.85,
        "severity": results.get("risk_level", "Unknown"),
        "tumor_area": results.get("metrics", {}).get("area", 0),
        "image_url": f"/static/outputs/{output_filename}",
        "findings": findings,
    }


# ========== UNIFIED PREDICTION API ==========
@app.route("/predict", methods=["POST"])
def predict():
    try:
        files = request.files.getlist("file")
        organ = request.form.get("organ", "").lower()
        pathology = request.form.get("pathology", "").lower()

        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No file uploaded"}), 400

        # Save all uploaded files
        saved_paths = []
        mhd_path = None
        for f in files:
            fpath = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(fpath)
            saved_paths.append(fpath)
            if f.filename.endswith(".mhd"):
                mhd_path = fpath

        # Use first file as main filepath (for non-.mhd uploads)
        filepath = saved_paths[0]

        result = None

        # ===== BRAIN PATHOLOGIES =====
        if organ == "brain":
            if BRAIN_AVAILABLE:
                try:
                    if "tumor" in pathology:
                        result = run_brain_tumor(filepath)
                    elif "hemorrhage" in pathology:
                        result = run_hemorrhage(filepath)
                    elif "stroke" in pathology:
                        result = run_stroke(filepath)
                except Exception as e:
                    print(f"Brain pipeline error: {e}")
                    traceback.print_exc()

        # ===== LUNG PATHOLOGIES =====
        elif organ == "lungs" or organ == "lung":
            scan_path = mhd_path or filepath
            if LUNG_AVAILABLE and scan_path.endswith(".mhd"):
                try:
                    result = _run_lung_pipeline(scan_path, pathology)
                except Exception as e:
                    print(f"Lung pipeline error: {e}")
                    traceback.print_exc()

        # ===== KIDNEY PATHOLOGIES =====
        elif organ == "kidney":
            if KIDNEY_AVAILABLE:
                try:
                    option = "tumor"
                    if "stone" in pathology:
                        option = "stone"
                    elif "cyst" in pathology:
                        option = "cyst"
                    elif "atrophy" in pathology:
                        option = "atrophy"
                    elif "hydronephrosis" in pathology:
                        option = "hydronephrosis"
                    result = run_kidney_analysis(filepath, option)
                except Exception as e:
                    print(f"Kidney pipeline error: {e}")
                    traceback.print_exc()

        # ===== LIVER PATHOLOGIES =====
        elif organ == "liver":
            try:
                result = _run_liver_pipeline(filepath, pathology)
            except Exception as e:
                print(f"Liver pipeline error: {e}")
                traceback.print_exc()

        # ===== FALLBACK =====
        if result is None:
            result = fallback_analysis(filepath, organ or "scan", pathology or "analysis")

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


# ========== LUNG SCAN BROWSER ==========


@app.route("/api/lungs/scans")
def list_lung_scans():
    """List available LUNA16 scans on disk"""
    try:
        annotations = pd.read_csv(ANNOTATIONS_PATH)
        available = []
        seen = set()
        for _, row in annotations.iterrows():
            uid = row.seriesuid
            if uid in seen:
                continue
            scan_path = os.path.join(LUNA16_PATH, f"{uid}.mhd")
            if os.path.exists(scan_path):
                seen.add(uid)
                available.append({
                    "id": uid,
                    "short_id": uid[-12:],
                    "diameter_mm": round(float(row.diameter_mm), 2)
                })
        return jsonify({"scans": available, "count": len(available)})
    except Exception as e:
        return jsonify({"error": str(e), "scans": [], "count": 0})


@app.route("/api/lungs/analyze", methods=["POST"])
def analyze_lung_scan():
    """Run full lung pipeline on a LUNA16 scan"""
    try:
        data = request.get_json()
        scan_id = data.get("scan_id", "")
        pathology = data.get("pathology", "nodule")

        scan_path = os.path.join(LUNA16_PATH, f"{scan_id}.mhd")
        if not os.path.exists(scan_path):
            return jsonify({"error": f"Scan not found: {scan_id}"}), 404

        # Load annotations for this scan
        annotations = pd.read_csv(ANNOTATIONS_PATH)
        scan_annotations = annotations[annotations.seriesuid == scan_id]
        if scan_annotations.empty:
            return jsonify({"error": "No annotations found for this scan"}), 404

        annotation = scan_annotations.iloc[0]

        if not LUNG_AVAILABLE:
            return jsonify({"error": "Lung module not available"}), 503

        # Run full pipeline (no plt.show)
        results = lung_analyzer.analyze(
            image_path=scan_path,
            pathology=pathology,
            annotation=annotation,
            visualize=False
        )

        # Save analysis image
        output_filename = f"lung_{pathology}_{int(time.time())}.png"
        save_path = os.path.join(OUTPUT_FOLDER, output_filename)

        analysis_img = results.get("analysis_img")
        if analysis_img is not None:
            # analysis_img is RGB, cv2.imwrite expects BGR
            cv2.imwrite(save_path, cv2.cvtColor(analysis_img, cv2.COLOR_RGB2BGR))

        # Build findings dict (convert numpy types to native Python)
        findings = {}
        for k, v in results.get("findings", {}).items():
            findings[k] = str(v)

        return jsonify({
            "detection": f"{results['pathology_name']} Detected",
            "confidence": 0.85,
            "severity": results.get("risk_level", "Unknown"),
            "tumor_area": results.get("metrics", {}).get("area", 0),
            "image_url": f"/static/outputs/{output_filename}",
            "findings": findings,
            "risk_level": results.get("risk_level", "Unknown"),
            "slice_index": results.get("slice_index", 0),
            "volume_shape": list(results.get("volume_shape", []))
        })

    except Exception as e:
        print(f"ERROR in lung analyze: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Lung analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)