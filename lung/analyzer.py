"""
Central Router for Pathology Analysis
This is what your website calls when a user selects a pathology

Usage:
    analyzer = PathologyAnalyzer()
    results = analyzer.analyze(
        image_path='luna16/subset0/scan.mhd',
        pathology='nodule',
        annotation=annotation_row
    )
"""

import os
import numpy as np
import pandas as pd

from shared.image_utils import load_medical_image
from shared.segmentation import segment_with_bbox
from shared.mask_utils import refine_mask, compute_mask_metrics
from shared.visualization import visualize_pathology_analysis, draw_analysis_overlay

from lung.pathologies import get_pathology, get_available_pathologies


class PathologyAnalyzer:
    """
    Main analysis router for different lung pathologies
    Coordinates the complete analysis pipeline
    """
    
    def __init__(self):
        self.available_pathologies = get_available_pathologies()
        print(f"Available pathologies: {self.available_pathologies}")
    
    def get_pathology_options(self):
        """Return list of available pathologies for UI display"""
        return {
            'nodule': 'Lung Nodule',
            'pneumonia': 'Pneumonia/Consolidation',
            'fibrosis': 'Pulmonary Fibrosis',
        }
    
    def analyze(self, image_path, pathology, annotation, visualize=True):
        """
        Run analysis for a selected pathology
        
        Args:
            image_path: Path to medical image (.mhd file)
            pathology: Selected pathology name ('nodule', 'pneumonia', 'fibrosis')
            annotation: Annotation data (varies by pathology type)
            visualize: Whether to display visualizations
        
        Returns:
            results: Dictionary with analysis results
        
        Raises:
            ValueError: If pathology not found
        """
        # Get pathology instance
        pathology_pipeline = get_pathology(pathology)
        if not pathology_pipeline:
            raise ValueError(f"Unknown pathology: {pathology}")
        
        print(f"\n{'='*50}")
        print(f"ANALYZING: {pathology_pipeline.name}")
        print(f"{'='*50}")
        
        # Step 1: Load medical image
        print("\n[Step 1] Loading medical image...")
        volume, origin, spacing = load_medical_image(image_path)
        print(f"Volume shape: {volume.shape}")
        
        # Step 2: Extract region of interest (pathology-specific)
        print(f"\n[Step 2] Extracting {pathology_pipeline.name} region...")
        slice_img, slice_index, center_coords = pathology_pipeline.extract_region_of_interest(
            volume, 
            annotation,
            origin=origin,
            spacing=spacing
        )
        print(f"Extracted slice {slice_index} at region {center_coords}")
        
        # Step 3: Prepare segmentation input (pathology-specific)
        print(f"\n[Step 3] Preparing segmentation input...")
        # Get diameter if available (for nodule)
        diameter_mm = getattr(annotation, 'diameter_mm', None)
        
        if pathology == 'nodule':
            prepared_img, bbox = pathology_pipeline.prepare_segmentation_input(
                slice_img, 
                center_coords,
                diameter_mm=diameter_mm
            )
        else:
            prepared_img, bbox = pathology_pipeline.prepare_segmentation_input(
                slice_img, 
                center_coords
            )
        print(f"Bounding box: {bbox}")
        
        # Step 4: Segment with SAM
        print(f"\n[Step 4] Running SAM segmentation...")
        mask = segment_with_bbox(prepared_img, bbox)
        print(f"Segmentation mask shape: {mask.shape}")
        
        # Step 5: Refine mask
        print(f"\n[Step 5] Refining segmentation mask...")
        refined_mask = refine_mask(mask)
        
        # Step 6: Compute general metrics
        print(f"\n[Step 6] Computing metrics...")
        metrics = compute_mask_metrics(refined_mask)
        print(f"Mask metrics: {metrics}")
        
        # Step 7: Compute pathology-specific findings
        print(f"\n[Step 7] Computing {pathology_pipeline.name} findings...")
        findings = pathology_pipeline.compute_findings(refined_mask, metrics)
        print(f"Findings: {findings}")
        
        # Get risk assessment
        risk = pathology_pipeline.get_risk_assessment()
        
        # Step 8: Create analysis overlay
        print(f"\n[Step 8] Creating analysis visualization...")
        analysis_info = {
            "Pathology": pathology_pipeline.name,
            "Risk Level": risk,
            **findings
        }
        analysis_img = draw_analysis_overlay(prepared_img, refined_mask, analysis_info)
        
        # Step 9: Visualize results
        if visualize:
            print(f"\n[Step 9] Displaying results...")
            visualize_pathology_analysis(
                prepared_img,
                bbox,
                refined_mask,
                analysis_info,
                title=f"{pathology_pipeline.name} Analysis"
            )
        
        # Compile results
        results = {
            'pathology': pathology,
            'pathology_name': pathology_pipeline.name,
            'slice_index': slice_index,
            'center_coords': center_coords,
            'bbox': bbox,
            'prepared_img': prepared_img,
            'mask': refined_mask,
            'risk_level': risk,
            'metrics': metrics,
            'findings': findings,
            'analysis_img': analysis_img,
            'volume_shape': volume.shape,
        }
        
        print(f"\n{'='*50}")
        print(f"ANALYSIS COMPLETE")
        print(f"Risk Level: {risk}")
        print(f"{'='*50}\n")
        
        return results


def run_example_analysis():
    """Example: Run analysis for demonstration"""
    
    # Load annotations
    annotations = pd.read_csv("lung/annotations.csv")
    subset0_path = "lung/luna16/subset0"
    
    # Find available scans
    available_rows = []
    for _, r in annotations.iterrows():
        series_uid = r.seriesuid
        path = f"{subset0_path}/{series_uid}.mhd"
        if os.path.exists(path):
            available_rows.append(r)
    
    if not available_rows:
        print("No annotated scans found!")
        return
    
    # Select a scan
    row = available_rows[0]  # Use first available
    series_uid = row.seriesuid
    ct_path = f"{subset0_path}/{series_uid}.mhd"
    
    # Create analyzer
    analyzer = PathologyAnalyzer()
    
    # Show available options
    print("\nAvailable pathologies:")
    for key, name in analyzer.get_pathology_options().items():
        print(f"  - {key}: {name}")
    
    # Run analysis for nodule (default)
    print(f"\nAnalyzing scan: {series_uid}")
    try:
        results = analyzer.analyze(
            image_path=ct_path,
            pathology='nodule',
            annotation=row,
            visualize=True
        )
        print(f"\nResults: Risk = {results['risk_level']}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_example_analysis()
