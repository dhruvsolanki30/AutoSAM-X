# Detection module
def detect_tumor(image):
    """
    Placeholder for YOLO-based tumor detection.
    Returns a bounding box.
    """
    height, width = image.shape

    # Dummy bounding box (center region)
    x1 = int(width * 0.3)
    y1 = int(height * 0.3)
    x2 = int(width * 0.7)
    y2 = int(height * 0.7)

    return [x1, y1, x2, y2]

