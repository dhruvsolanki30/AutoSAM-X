"""
Simple Multi-Pathology Lung Analysis
Select pathology → Add image → Detect boundary → Display results
"""

import pandas as pd
import numpy as np
import cv2
import os
import random
from modules.lung.analyzer import PathologyAnalyzer


def select_pathology():
    """Show options and let user select"""
    analyzer = PathologyAnalyzer()
    options = analyzer.get_pathology_options()
    
    print("\n" + "="*60)
    print("LUNG PATHOLOGY ANALYSIS - SELECT CONDITION")
    print("="*60)
    
    pathology_list = list(options.items())
    for i, (key, name) in enumerate(pathology_list, 1):
        print(f"{i}. {name} ({key})")
    
    while True:
        try:
            choice = int(input("\nSelect pathology (1-3): "))
            if 1 <= choice <= len(pathology_list):
                selected_key, selected_name = pathology_list[choice - 1]
                print(f"✓ Selected: {selected_name}")
                return selected_key, selected_name
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")


def select_scan():
    """Show available scans and let user select"""
    print("\n" + "="*60)
    print("SELECT SCAN")
    print("="*60)
    
    try:
        annotations = pd.read_csv("annotations.csv")
    except FileNotFoundError:
        print("✗ annotations.csv not found!")
        return None, None
    
    # Find available scans
    available_scans = []
    subset0_path = "luna16/subset0"
    
    for idx, row in annotations.iterrows():
        scan_path = f"{subset0_path}/{row.seriesuid}.mhd"
        if os.path.exists(scan_path):
            available_scans.append((idx, row, scan_path))
    
    if not available_scans:
        print("✗ No CT scans found in luna16/subset0/")
        return None, None
    
    print(f"Found {len(available_scans)} scans\n")
    print("Options:")
    print("1. Random scan")
    print("2. First available scan")
    print("3. Enter scan ID")
    
    choice = input("\nChoose (1-3): ").strip()
    
    if choice == "1":
        idx, row, path = random.choice(available_scans)
        print(f"✓ Selected random scan: {row.seriesuid}")
        return row, path
    elif choice == "2":
        idx, row, path = available_scans[0]
        print(f"✓ Selected: {row.seriesuid}")
        return row, path
    elif choice == "3":
        scan_id = input("Enter scan ID: ").strip()
        for idx, row, path in available_scans:
            if scan_id in row.seriesuid:
                print(f"✓ Found: {row.seriesuid}")
                return row, path
        print("✗ Scan not found")
        return None, None
    else:
        print("Invalid choice")
        return None, None


def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Multi-Pathology Lung Analysis System             ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # Step 1: Select pathology
    pathology, pathology_name = select_pathology()
    if not pathology:
        return
    
    # Step 2: Select scan
    annotation, scan_path = select_scan()
    if annotation is None:
        return
    
    # Step 3: Run analysis
    print("\n" + "="*60)
    print("ANALYZING...")
    print("="*60)
    print(f"Pathology: {pathology_name}")
    print(f"Scan: {annotation.seriesuid}")
    
    try:
        analyzer = PathologyAnalyzer()
        results = analyzer.analyze(
            image_path=scan_path,
            pathology=pathology,
            annotation=annotation,
            visualize=True  # Show plots
        )
        
        # Display summary
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Risk Level: {results['risk_level']}")
        print(f"\nFindings:")
        for key, value in results['findings'].items():
            print(f"  • {key}: {value}")
        
    except Exception as e:
        print(f"\n✗ Error during analysis:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
