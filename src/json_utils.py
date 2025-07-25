import json
import numpy as np

def numpy_to_python_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy data types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_types(item) for item in obj]
    else:
        return obj

def save_to_json(data, filepath):
    """
    Save data to JSON file, handling numpy data types.
    
    Args:
        data: Data to save (can contain numpy types)
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(numpy_to_python_types(data), f, indent=4)