"""
Lung pathology detection pipelines
Each pathology implements the PathologyPipeline interface
"""

from .nodule import LungNodulePathology
from .pneumonia import PneumoniaPathology
from .fibrosis import FibrosisPathology

# Registry of available pathologies
PATHOLOGY_REGISTRY = {
    'nodule': LungNodulePathology,
    'pneumonia': PneumoniaPathology,
    'fibrosis': FibrosisPathology,
}


def get_available_pathologies():
    """Return list of available pathologies"""
    return list(PATHOLOGY_REGISTRY.keys())


def get_pathology(pathology_name):
    """
    Get a pathology pipeline by name
    
    Args:
        pathology_name: Name of pathology ('nodule', 'pneumonia', 'fibrosis')
    
    Returns:
        Pathology instance or None if not found
    """
    PathologyClass = PATHOLOGY_REGISTRY.get(pathology_name)
    if PathologyClass:
        return PathologyClass()
    return None


__all__ = [
    'LungNodulePathology',
    'PneumoniaPathology',
    'FibrosisPathology',
    'PATHOLOGY_REGISTRY',
    'get_available_pathologies',
    'get_pathology'
]
