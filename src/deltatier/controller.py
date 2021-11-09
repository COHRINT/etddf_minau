import numpy as np

def scan_control(position_dict, prototracks, scan_size, uncertainty_thresh):
    """
    Returns the lower & upper scan regions 
    """
    # Called when we complete a scan of the target region

    # This is a simple memoryless function, some memory of previously scanned agent