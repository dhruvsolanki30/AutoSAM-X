"""
Base class for medical pathology detection pipelines
Provides the framework that each specific pathology implements
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any


class PathologyPipeline(ABC):
    """
    Abstract base class for all pathology detection pipelines
    Each specific pathology (Nodule, Pneumonia, Fibrosis) inherits from this
    """
    
    def __init__(self, name: str, description: str):
        """
        Args:
            name: Pathology name (e.g., "Lung Nodule")
            description: Pathology description
        """
        self.name = name
        self.description = description
        self.metrics = {}
        self.risk_level = None
    
    @abstractmethod
    def extract_region_of_interest(self, volume, annotation, origin=None, spacing=None) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Extract the 2D slice and coordinates from 3D volume
        
        Args:
            volume: 3D medical image volume
            annotation: Annotation data (varies by pathology)
            origin: Image origin coordinates
            spacing: Voxel spacing
        
        Returns:
            slice_img: 2D extracted slice
            slice_index: Which slice was extracted
            center_coords: (center_x, center_y) in the slice
        """
        pass
    
    @abstractmethod
    def prepare_segmentation_input(self, slice_img: np.ndarray, center_coords: Tuple[int, int]) -> Tuple[np.ndarray, list]:
        """
        Prepare image and bounding box for SAM segmentation
        
        Args:
            slice_img: 2D slice
            center_coords: Center coordinates in original slice
        
        Returns:
            prepared_img: Image ready for SAM (RGB, 512x512)
            bbox: Bounding box [x1, y1, x2, y2]
        """
        pass
    
    @abstractmethod
    def compute_findings(self, mask: np.ndarray, metrics_dict: Dict) -> Dict[str, Any]:
        """
        Analyze the segmented mask and compute pathology-specific metrics
        
        Args:
            mask: Segmentation mask from SAM
            metrics_dict: Basic metrics (area, perimeter, circularity, centroid)
        
        Returns:
            findings: Dict with diagnostic findings and severity assessment
        """
        pass
    
    @abstractmethod
    def get_risk_assessment(self) -> str:
        """
        Return risk level text for this pathology
        
        Returns:
            risk_text: String describing risk (e.g., "LOW", "MEDIUM", "HIGH")
        """
        pass
    
    def get_summary(self) -> Dict[str, str]:
        """Get summary of findings for display"""
        return {
            "Pathology": self.name,
            "Risk": self.get_risk_assessment() if self.metrics else "N/A",
            **self.metrics
        }
