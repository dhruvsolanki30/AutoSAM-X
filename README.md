# AutoSAM-X

**Automatic Multi-Organ Medical Image Segmentation using YOLO + SAM**

AutoSAM-X is a fully automatic, multi-organ medical image analysis framework that combines YOLO-based object detection with the Segment Anything Model (SAM) to deliver zero-interaction segmentation and diagnosis from CT/MRI scans.

## Supported Organs & Pathologies

| Organ | Pathologies | Input Formats |
|-------|------------|---------------|
| **Brain** | Tumor, Hemorrhage, Stroke | JPEG, PNG |
| **Kidney** | Tumor, Stone, Cyst, Atrophy, Hydronephrosis | NIfTI (.nii), JPEG, PNG |
| **Liver** | Tumor, Cirrhosis | NIfTI (.nii/.nii.gz), JPEG, PNG |
| **Lung** | Nodule, Pneumonia, Fibrosis | MetaImage (.mhd + .raw), JPEG, PNG |

## How It Works

1. **Upload** a medical scan through the web interface
2. **Select** the organ and pathology to analyze
3. **View** a 3-panel result: Original → Detection (bounding box) → Segmentation (mask overlay)

The pipeline preprocesses the input, detects regions of interest using YOLO or intensity-based methods, segments with SAM, and classifies the pathology.

## Project Structure

```
AutoSAM-X/
├── app.py              # Flask web server
├── brain/              # Brain analysis (tumor, hemorrhage, stroke)
├── kidney/             # Kidney analysis (tumor, stone, cyst, atrophy, hydronephrosis)
├── liver/              # Liver analysis (tumor, cirrhosis)
├── lung/               # Lung analysis (nodule, pneumonia, fibrosis)
├── shared/             # Shared utilities (SAM, masks, visualization)
├── templates/          # HTML templates
├── static/             # UI assets, uploads, outputs
├── sam_vit_b.pth       # SAM ViT-B weights (download separately)
└── requirements.txt
```

## Setup

```bash
# Clone
git clone https://github.com/dhruvsolanki30/AutoSAM-X.git
cd AutoSAM-X

# Install dependencies
pip install -r requirements.txt

# Download SAM weights (~375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth

# Run
python app.py
```

Open `http://localhost:5000` in your browser.

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Segmentation:** Segment Anything Model (SAM ViT-B)
- **Refinement:** U-Net (MONAI), morphological post-processing
- **Backend:** Flask
- **Medical Imaging:** nibabel (NIfTI), SimpleITK (MetaImage)
