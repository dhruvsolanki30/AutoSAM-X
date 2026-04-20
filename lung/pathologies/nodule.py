"""
Lung Nodule Detection Pipeline
Detects and segments suspicious nodules in lung CT scans
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Any

from lung.pathology_base import PathologyPipeline
from shared.image_utils import (
    world_to_voxel, extract_slice, resize_and_convert,
    scale_coordinates, create_bounding_box
)


class LungNodulePathology(PathologyPipeline):
    """
    Lung Nodule detection using annotations with center coordinates and diameter
    """
    
    def __init__(self):
        super().__init__(
            name="Lung Nodule",
            description="Suspicious lung nodules that may indicate malignancy"
        )
    
    def extract_region_of_interest(self, volume, annotation, origin=None, spacing=None) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Extract nodule region from volume using center coordinates
        
        Args:
            volume: 3D CT volume
            annotation: Row with coordZ, coordY, coordX (world coords)
            origin: Image origin
            spacing: Voxel spacing
        
        Returns:
            slice_img: 2D grayscale slice
            slice_index: Z-index
            center_coords: (center_x, center_y) in 2D slice
        """
        # Get world coordinates from annotation
        coord_world = np.array([annotation.coordZ, annotation.coordY, annotation.coordX])
        
        # Convert world coordinates to voxel coordinates
        if origin is not None and spacing is not None:
            origin_rev = origin[::-1]  # Reverse for [Z, Y, X] order
            spacing_rev = spacing[::-1]
            voxel_coord = world_to_voxel(coord_world, origin_rev, spacing_rev)
        else:
            # Fallback: assume world coords are already voxel coords
            voxel_coord = coord_world.astype(int)
        
        slice_index = voxel_coord[0]
        slice_img = extract_slice(volume, slice_index, normalize=True)
        
        center_x = voxel_coord[2]
        center_y = voxel_coord[1]
        
        return slice_img, slice_index, (center_x, center_y)
    
    def prepare_segmentation_input(self, slice_img: np.ndarray, center_coords: Tuple[int, int], 
                                   diameter_mm: float = None) -> Tuple[np.ndarray, list]:
        """
        Prepare image for SAM segmentation
        
        Args:
            slice_img: Original 2D slice
            center_coords: Center point in original slice
            diameter_mm: Nodule diameter in mm (for bbox size)
        
        Returns:
            prepared_img: RGB image resized to 512x512
            bbox: Bounding box in 512x512 space
        """
        # Resize to 512x512
        resized_img, original_shape = resize_and_convert(slice_img, target_size=512)
        
        # Scale center coordinates to 512x512 space
        center = np.array(center_coords)
        scaled_center = scale_coordinates(center, original_shape, target_size=512)
        center_x, center_y = scaled_center
        
        # Create bounding box (diameter in mm -> pixels)
        # Typical nodule radius ~10-20 pixels at 512x512
        radius = int(diameter_mm * 2) if diameter_mm else 20
        bbox = create_bounding_box(center_x, center_y, radius, img_size=512)
        
        return resized_img, bbox
    
    def compute_findings(self, mask: np.ndarray, metrics_dict: Dict) -> Dict[str, Any]:
        """
        Compute nodule-specific findings
        
        Args:
            mask: Segmentation mask
            metrics_dict: Basic metrics (area, perimeter, circularity, centroid)
        
        Returns:
            findings: Diagnostic metrics
        """
        area = metrics_dict.get('area', 0)
        circularity = metrics_dict.get('circularity', 0)
        
        findings = {
            "Area (pixels)": area,
            "Circularity": f"{circularity:.2f}",
        }
        
        # Estimate diameter from area (rough approximation)
        if area > 0:
            estimated_diameter = 2 * np.sqrt(area / np.pi)
            findings["Est. Diameter (px)"] = f"{estimated_diameter:.1f}"
        
        self.metrics = findings
        return findings
    
    def get_risk_assessment(self) -> str:
        """
        Assess nodule risk based on size
        - LOW: < 6mm
        - MEDIUM: 6-10mm
        - HIGH: > 10mm
        """
        if not self.metrics:
            return "UNKNOWN"
        
        est_diameter_str = self.metrics.get("Est. Diameter (px)", "0")
        try:
            est_diameter = float(est_diameter_str)
        except:
            return "UNKNOWN"
        
        if est_diameter < 6:
            self.risk_level = "LOW"
        elif est_diameter < 10:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"
        
        return self.risk_level
