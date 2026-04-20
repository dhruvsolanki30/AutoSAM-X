import numpy as np

def classify(slice_img, base_mask, option):

    result = {}
    disease_mask = np.zeros_like(slice_img)

    if option == "tumor":

        disease_mask = base_mask
        size = int(np.sum(base_mask))

        result["Diagnosis"] = "Tumor Detected" if size > 500 else "Healthy"
        result["Tumor_Size"] = size

    elif option == "stone":

        disease_mask = (slice_img > 200).astype(np.uint8)
        count = int(np.sum(disease_mask))

        result["Diagnosis"] = "Kidney Stones Detected" if count > 50 else "Healthy"
        result["Stone_Count"] = count

    elif option == "cyst":

        disease_mask = (slice_img < 40).astype(np.uint8)
        pixels = int(np.sum(disease_mask))

        result["Diagnosis"] = "Kidney Cyst Detected" if pixels > 500 else "Healthy"

    elif option == "atrophy":

        disease_mask = base_mask
        area = int(np.sum(base_mask))

        result["Diagnosis"] = "Kidney Atrophy" if area < 2000 else "Healthy"
        result["Kidney_Size"] = area

    elif option == "hydronephrosis":

        disease_mask = (slice_img < 30).astype(np.uint8)
        area = int(np.sum(disease_mask))

        result["Diagnosis"] = "Hydronephrosis" if area > 1000 else "Healthy"

    return result, disease_mask