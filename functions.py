



def detector_matrix(data_dict, filter_labels, pixel):
    """
    Create a matrix from the data dictionary, filter by labels, and normalize coordinates.

    Parameters:
    - data_dict: Dictionary containing data with y, z coordinates and labels.
    - filter_labels: Set of labels to filter by.
    - size: Number of (rows,columns) in the matrix.
    
    Returns:
    - Realistic matrix of the detector
    """
    
    import math
    import numpy as np

    size = (math.ceil(400/pixel[0]), math.ceil(500/pixel[1]))
   
    # Extract y and z values and the corresponding indices from the dictionary
    y_values = [val[1] for val in data_dict.values()]
    z_values = [val[2] for val in data_dict.values()]
    labels = [val[3] for val in data_dict.values()]
    indices = list(data_dict.keys())

    # Normalize and scale the y and z data to fit into the matrix (tengo que cambiarlo para poner maximo y minimo de la caja)
    y_min,y_max, z_min, z_max  = -200, 200, 0, 500

    y_scaled = [(y - y_min) / (y_max - y_min) * (size[0] - 1) for y in y_values]
    z_scaled = [(z - z_min) / (z_max - z_min) * (size[1] - 1) for z in z_values]

    # Rounding to the nearest integer for matrix indices
    y_indices = [round(y) for y in y_scaled]
    z_indices = [round(z) for z in z_scaled]

    # Create the matrix and initialize it with None or 0
    matrix = np.full(size, -2)

    # Populate the matrix with the indices from the dictionary

    for idx, (y, z) in enumerate(zip(y_indices, z_indices)):
        if labels[idx] in filter_labels:
            matrix[y, z] = indices[idx]

    return matrix