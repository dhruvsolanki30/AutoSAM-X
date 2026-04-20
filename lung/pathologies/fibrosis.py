"""
Pulmonary Fibrosis Detection Pipeline
Identifies fibrotic regions in lung CT scans
"""
import numpy as np
from typing import Dict, Tuple, Any

from lung.pathology_base import PathologyPipeline
from shared.image_utils import (
    extract_slice, resize_and_convert, scale_coordinates, create_bounding_box
)


class FibrosisPathology(PathologyPipeline):
    """
    Pulmonary Fibrosis detection - identifies scarring/fibrotic tissue
    Typically appears as linear/reticular patterns in lower lung lobes
    """
    
    def __init__(self):
        super().__init__(
            name="Pulmonary Fibrosis",
            description="Fibrotic scarring and remodeling indicating lung fibrosis"
        )
    
    def extract_region_of_interest(self, volume, annotation, origin=None, spacing=None) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Extract fibrosis region from volume
        
        Args:
            volume: 3D CT volume
            annotation: Dict or object with fibrosis location info
            origin: Image origin (unused for fibrosis)
            spacing: Voxel spacing (unused for fibrosis)
        
        Returns:
            slice_img: 2D grayscale slice
            slice_index: Z-index
            center_coords: (center_x, center_y)
        """
        if hasattr(annotation, 'slice_index'):
            slice_index = annotation.slice_index
            center_x = annotation.center_x
            center_y = annotation.center_y
        else:
            slice_index = annotation.get('slice_index', volume.shape[0] // 2)
            center_x = annotation.get('center_x', 256)
            center_y = annotation.get('center_y', 256)
        
        slice_img = extract_slice(volume, slice_index, normalize=True)
        
        return slice_img, slice_index, (center_x, center_y)
    
    def prepare_segmentation_input(self, slice_img: np.ndarray, center_coords: Tuple[int, int],
                                   region_size: float = 50) -> Tuple[np.ndarray, list]:
        """
        Prepare image for SAM segmentation
        
        Args:
            slice_img: Original 2D slice
            center_coords: Center of fibrosis region
            region_size: Size of region to analyze
        
        Returns:
            prepared_img: RGB image resized to 512x512
            bbox: Bounding box for fibrotic region
        """
        # Resize to 512x512
        resized_img, original_shape = resize_and_convert(slice_img, target_size=512)
        
        # Scale center
        center = np.array(center_coords)
        scaled_center = scale_coordinates(center, original_shape, target_size=512)
        center_x, center_y = scaled_center
        
        # Fibrosis often involves larger areas, so bigger bbox
        radius = int(region_size * 2) if region_size else 50
        bbox = create_bounding_box(center_x, center_y, radius, img_size=512)
        
        return resized_img, bbox
    
    def compute_findings(self, mask: np.ndarray, metrics_dict: Dict) -> Dict[str, Any]:
        """
        Compute fibrosis-specific findings
        
        Args:
            mask: Segmentation mask of fibrotic tissue
            metrics_dict: Basic metrics
        
        Returns:
            findings: Fibrosis-specific metrics
        """
        area = metrics_dict.get('area', 0)
        circularity = metrics_dict.get('circularity', 0)
        
        findings = {
            "Fibrotic Area (px)": area,
            "Texture Regularity": f"{circularity:.2f}",
        }
        
        # Assess fibrosis severity based on area
        if area < 2000:
            severity = "Mild"
        elif area < 8000:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        findings["Severity"] = severity
        
        # Texture indicates pattern - lower circularity = more reticular
        if circularity < 0.4:
            pattern = "Reticular (lace-like)"
        elif circularity < 0.7:
            pattern = "Mixed"
        else:
            pattern = "Nodular"
        
        findings["Pattern"] = pattern
        
        self.metrics = findings
        return findings
    
    def get_risk_assessment(self) -> str:
        """
        Assess fibrosis risk/severity
        - LOW: Mild fibrosis
        - MEDIUM: Moderate fibrosis
        - HIGH: Severe fibrosis with extensive involvement
        """
        if not self.metrics:
            return "UNKNOWN"
        
        severity = self.metrics.get("Severity", "Unknown")
        
        if severity == "Mild":
            self.risk_level = "LOW"
        elif severity == "Moderate":
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"
        
        return self.risk_level
