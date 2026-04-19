"""
Pneumonia Detection Pipeline
Identifies regions of consolidation/infiltrate in lung CT scans
"""
import numpy as np
from typing import Dict, Tuple, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared'))

from pathology_base import PathologyPipeline
from image_utils import (
    extract_slice, resize_and_convert, scale_coordinates, create_bounding_box
)


class PneumoniaPathology(PathologyPipeline):
    """
    Pneumonia detection - identifies consolidation in lungs
    Uses center point and search radius for region selection
    """
    
    def __init__(self):
        super().__init__(
            name="Pneumonia/Consolidation",
            description="Regions of lung consolidation indicating pneumonia or infection"
        )
        self.search_radius = 15  # Default search radius in pixels
    
    def extract_region_of_interest(self, volume, annotation, origin=None, spacing=None) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Extract consolidation region from volume
        
        Args:
            volume: 3D CT volume
            annotation: Dict or object with slice_index, center_x, center_y
            origin: Image origin (unused for pneumonia)
            spacing: Voxel spacing (unused for pneumonia)
        
        Returns:
            slice_img: 2D grayscale slice
            slice_index: Z-index
            center_coords: (center_x, center_y)
        """
        # Handle different annotation formats
        if hasattr(annotation, 'slice_index'):
            slice_index = annotation.slice_index
            center_x = annotation.center_x
            center_y = annotation.center_y
        else:
            # Assume it's a dict-like object
            slice_index = annotation.get('slice_index', 0)
            center_x = annotation.get('center_x', 256)
            center_y = annotation.get('center_y', 256)
        
        # Extract and normalize
        slice_img = extract_slice(volume, slice_index, normalize=True)
        
        return slice_img, slice_index, (center_x, center_y)
    
    def prepare_segmentation_input(self, slice_img: np.ndarray, center_coords: Tuple[int, int],
                                   search_radius: float = 30) -> Tuple[np.ndarray, list]:
        """
        Prepare image for SAM segmentation
        
        Args:
            slice_img: Original 2D slice
            center_coords: Center of suspected consolidation
            search_radius: Search radius in mm or pixels
        
        Returns:
            prepared_img: RGB image resized to 512x512
            bbox: Bounding box for consolidation region
        """
        # Resize to 512x512
        resized_img, original_shape = resize_and_convert(slice_img, target_size=512)
        
        # Scale center to 512x512
        center = np.array(center_coords)
        scaled_center = scale_coordinates(center, original_shape, target_size=512)
        center_x, center_y = scaled_center
        
        # Create bounding box (consolidations can be larger ~40-50px)
        radius = int(search_radius * 2.5) if search_radius else 40
        bbox = create_bounding_box(center_x, center_y, radius, img_size=512)
        
        return resized_img, bbox
    
    def compute_findings(self, mask: np.ndarray, metrics_dict: Dict) -> Dict[str, Any]:
        """
        Compute pneumonia-specific findings
        
        Args:
            mask: Segmentation mask of consolidation
            metrics_dict: Basic metrics
        
        Returns:
            findings: Pneumonia-specific metrics
        """
        area = metrics_dict.get('area', 0)
        perimeter = metrics_dict.get('perimeter', 0)
        
        findings = {
            "Consolidation Area (px)": area,
            "Perimeter (px)": f"{perimeter:.1f}",
        }
        
        # Estimate extent of consolidation
        if area < 1000:
            extent = "Small"
        elif area < 5000:
            extent = "Moderate"
        else:
            extent = "Large"
        
        findings["Extent"] = extent
        
        self.metrics = findings
        return findings
    
    def get_risk_assessment(self) -> str:
        """
        Assess consolidation severity
        - LOW: Small consolidation
        - MEDIUM: Moderate consolidation
        - HIGH: Large/extensive consolidation
        """
        if not self.metrics:
            return "UNKNOWN"
        
        extent = self.metrics.get("Extent", "Unknown")
        
        if extent == "Small":
            self.risk_level = "LOW"
        elif extent == "Moderate":
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"
        
        return self.risk_level
