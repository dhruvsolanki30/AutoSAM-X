"""
Shared image preprocessing utilities for all pathologies
"""
import SimpleITK as sitk
import numpy as np
import cv2


def load_medical_image(image_path):
    """
    Load 3D medical imaging data (MetaImage, NIfTI, etc.)
    
    Args:
        image_path: Path to .mhd, .nii, or other medical imaging format
    
    Returns:
        volume: 3D numpy array
        origin: Origin coordinates in physical space
        spacing: Voxel spacing in physical space
    """
    print(f"Loading medical image: {image_path}")
    
    image = sitk.ReadImage(image_path)
    volume = sitk.GetArrayFromImage(image)
    
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    
    print(f"Volume shape: {volume.shape}")
    print(f"Origin: {origin}, Spacing: {spacing}")
    
    return volume, origin, spacing


def world_to_voxel(world_coords, origin, spacing):
    """
    Convert world coordinates to voxel indices
    
    Args:
        world_coords: Coordinates in physical space (numpy array)
        origin: Image origin
        spacing: Voxel spacing
    
    Returns:
        voxel_coords: Voxel indices (numpy array)
    """
    voxel = (world_coords - origin) / spacing
    return voxel.astype(int)


def extract_slice(volume, slice_index, normalize=True):
    """
    Extract a 2D slice from 3D volume and optionally normalize
    
    Args:
        volume: 3D numpy array
        slice_index: Index of slice to extract
        normalize: If True, normalize to 0-255
    
    Returns:
        slice_img: 2D numpy array (uint8)
    """
    slice_img = volume[slice_index]
    
    if normalize:
        # Normalize to 0-1
        slice_min = np.min(slice_img)
        slice_max = np.max(slice_img)
        
        if slice_max > slice_min:
            slice_img = (slice_img - slice_min) / (slice_max - slice_min)
        else:
            slice_img = np.zeros_like(slice_img)
        
        # Scale to 0-255
        slice_img = (slice_img * 255).astype(np.uint8)
    else:
        slice_img = slice_img.astype(np.uint8)
    
    return slice_img


def resize_and_convert(image, target_size=512):
    """
    Resize image to target size and convert to RGB if grayscale
    
    Args:
        image: Input image (2D)
        target_size: Target resolution
    
    Returns:
        resized_img: RGB image
        original_shape: Original (height, width)
    """
    original_h, original_w = image.shape[:2]
    
    resized_img = cv2.resize(image, (target_size, target_size))
    
    if len(resized_img.shape) == 2:  # Grayscale
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    
    return resized_img, (original_h, original_w)


def scale_coordinates(coords, original_shape, target_size=512):
    """
    Scale coordinates from original image size to target (resized) size
    
    Args:
        coords: Original coordinates (x, y)
        original_shape: Original (height, width)
        target_size: Target resolution
    
    Returns:
        scaled_coords: Scaled coordinates
    """
    original_h, original_w = original_shape
    scale_x = target_size / original_w
    scale_y = target_size / original_h
    
    scaled_coords = coords.copy()
    scaled_coords[0] = int(scaled_coords[0] * scale_x)
    scaled_coords[1] = int(scaled_coords[1] * scale_y)
    
    return scaled_coords


def create_bounding_box(center_x, center_y, radius, img_size=512):
    """
    Create bounding box around center point with radius
    
    Args:
        center_x, center_y: Center of box
        radius: Half-width/half-height of box
        img_size: Image size (to clip to boundaries)
    
    Returns:
        bbox: [x1, y1, x2, y2]
    """
    bbox = [
        max(center_x - radius, 0),
        max(center_y - radius, 0),
        min(center_x + radius, img_size),
        min(center_y + radius, img_size)
    ]
    return bbox
