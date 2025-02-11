import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

import numpy as np

def image_creator_gen(pe_matrix, time_matrix, *maps):
    """
    Generalized function to create an image based on an arbitrary number of maps.
    
    :param pe_matrix: A 2D or 3D array of photoelectron counts.
    :param time_matrix: A 2D or 3D array of time values.
    :param maps: Any number of maps used for spatial distributions (vis_map, vuv_map, etc.).
    :return: An image array with channels corresponding to each map.
    """
    ch_z, ch_y = maps[0].shape  # Assuming all maps have the same shape
    n_events = pe_matrix.shape[0]

    # Create empty arrays to hold the results for each map
    map_count = len(maps)
    
    pe_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]

    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]

        #NEW NORMALIZATION

        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])
            time_matrices_map[idx] /= np.max(time_matrices_map[:2])
        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])
            time_matrices_map[idx] /= np.max(time_matrices_map[2:])

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count * 2))


    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        image[:, :, :, channel] = pe_mat[0]
        image[:, :, :, channel + 1] = pe_mat[1]
        image[:, :, :, channel + 2] = time_mat[0]
        image[:, :, :, channel + 3] = time_mat[1]
        channel += 4

    return image

def image_creator_gen2(pe_matrix, time_matrix, *maps):
    """
    Generalized function to create an image based on an arbitrary number of maps.
    
    :param pe_matrix: A 2D or 3D array of photoelectron counts.
    :param time_matrix: A 2D or 3D array of time values.
    :param maps: Any number of maps used for spatial distributions (vis_map, vuv_map, etc.).
    :return: An image array with channels corresponding to each map.
    """
    ch_z, ch_y = maps[0].shape  # Assuming all maps have the same shape
    n_events = pe_matrix.shape[0]

    # Create empty arrays to hold the results for each map
    map_count = len(maps)
    
    pe_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
       # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]

        #NEW NORMALIZATION

        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])
            time_matrices_map[idx] /= np.max(time_matrices_map[:2])
        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])
            time_matrices_map[idx] /= np.max(time_matrices_map[2:])

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    pe_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))
    time_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))  
    
    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        pe_image[:, :, :, channel] = pe_mat[0]
        pe_image[:, :, :, channel + 1] = pe_mat[1]
        time_image[:, :, :, channel] = time_mat[0]
        time_image[:, :, :, channel + 1] = time_mat[1]
        channel += 2

    return pe_image, time_image


def image_creator_gen3(pe_matrix, time_matrix, *maps):
    """
    Generalized function to create an image based on an arbitrary number of maps.
    
    :param pe_matrix: A 2D or 3D array of photoelectron counts.
    :param time_matrix: A 2D or 3D array of time values.
    :param maps: Any number of maps used for spatial distributions (vis_map, vuv_map, etc.).
    :return: An image array with channels corresponding to each map.
    """
    ch_z, ch_y = maps[0].shape  # Assuming all maps have the same shape
    n_events = pe_matrix.shape[0]

    # Create empty arrays to hold the results for each map
    map_count = len(maps)
    
    pe_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Get max time for this event
            max_time = np.max(time_matrices_map[idx][i])  
            
            # Get min nonzero time, but check if there are any nonzero values first
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0  # Avoid error
            
            # Apply transformation while ensuring zeros don't distort the range
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                max_time - time_matrices_map[idx][i] + min_time,  # Shift values
                0  # Ensure zeros remain zeros
            )
    
        pe_matrices_map[idx] /= np.max(pe_matrices_map[idx])  # Normalizar cada mapa de PE individualmente
    
    # Compute the global max across all time matrices
    global_max_time = max(np.max(time_mat) for time_mat in time_matrices_map_inv)

    # Normalize using the global max
    for idx in range(len(time_matrices_map_inv)):
        time_matrices_map_inv[idx] /= global_max_time

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count * 2))
    pe_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))
    time_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))  

    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        pe_image[:, :, :, channel] = pe_mat[0]
        pe_image[:, :, :, channel + 1] = pe_mat[1]
        time_image[:, :, :, channel] = time_mat[0]
        time_image[:, :, :, channel + 1] = time_mat[1]
        channel += 2

    return pe_image, time_image


def select_non_empty_half(matrix, method="max"):
    """
    Selects the half of the matrix (left or right) that contains more meaningful data.

    The selection is based on a specified method:
    - "max": Chooses the half with the highest maximum value.
    - "sum": Chooses the half with the highest total sum of values.
    - "nonzero": Chooses the half with the most nonzero elements.
    - "mean_top": Chooses the half with the highest mean of the top N values.

    Parameters:
    -----------
    matrix : np.ndarray
        The input 2D or 3D matrix (e.g., (n_events, height, width)).
    method : str, optional
        The selection criterion ("max", "sum", "nonzero", "mean_top"), by default "max".

    Returns:
    --------
    np.ndarray
        The selected half of the matrix.
    """

    # Ensure the matrix has an even number of columns (width)
    if matrix.shape[1] % 2 != 0:
        raise ValueError("The matrix width must be evenly divisible by 2 to split it equally.")

    # Split matrix into left and right halves
    left_half, right_half = np.hsplit(matrix, 2)

    if method == "max":
        left_score = np.max(left_half)
        right_score = np.max(right_half)
    elif method == "sum":
        left_score = np.sum(left_half)
        right_score = np.sum(right_half)
    elif method == "nonzero":
        left_score = np.count_nonzero(left_half)
        right_score = np.count_nonzero(right_half)
    elif method == "mean_top":
        top_n = 5  # Number of top values to consider
        left_score = np.mean(np.sort(left_half.flatten())[-top_n:])
        right_score = np.mean(np.sort(right_half.flatten())[-top_n:])
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from 'max', 'sum', 'nonzero', or 'mean_top'.")

    return left_half if left_score >= right_score else right_half


def plot_image(image_data, event_idx, labels, groups, grid, figsize=(26, 10), use_log_scale=False, show_colorbar=False):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int
        Index of the event to plot
    labels : list of str
        Labels for each subplot
    groups : list of list of int
        Grouping of the image indices for shared scaling
    grid : tuple
        Tuple specifying the number of rows and columns (num_rows, num_columns)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (26, 10))
    use_log_scale : bool, optional
        Whether to use logarithmic color scaling (default: False)
    show_colorbar : bool, optional
        Whether to show colorbars for each subplot (default: False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axs : numpy.ndarray
        Array of subplot axes
    """
    
    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define grid dimensions
    num_rows, num_columns = grid

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()
    
    # Set up colormap
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white')
    
    # Calculate scales for each group
    group_scales = {}
    for group in groups:
        relevant_images = [np.squeeze(image_data[event_index[0], :, :, i]) for i in group]
        if use_log_scale:
            vmin = 1e-10  # Small positive number for log scale
        else:
            vmin = 0
        vmax = max(np.max(img[img > 0]) if np.any(img > 0) else vmin for img in relevant_images)
        group_scales[tuple(group)] = (vmin, vmax)
    
    # Plot images
    for idx in range(len(labels)):
        img = np.squeeze(image_data[event_index[0], :, :, idx])
        masked_img = np.ma.masked_where(img <= 0, img)
        
        for group, (vmin, vmax) in group_scales.items():
            if idx in group:
                if use_log_scale:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       vmin=vmin, vmax=vmax)
                break
        
        axs[idx].set_title(labels[idx], fontsize=20)
        
        # Add colorbar if requested
        if show_colorbar:
            divider = make_axes_locatable(axs[idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    # Remove ticks
    plt.setp(axs, xticks=[], yticks=[])
    
    plt.tight_layout()
    
    plt.show()

def plot_image2(image_data, event_idx, labels, groups, grid, figsize=(26, 10), use_log_scale=False, show_colorbar=True):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int
        Index of the event to plot
    labels : list of str
        Labels for each subplot
    groups : list of list of int
        Grouping of the image indices for shared scaling
    grid : tuple
        Tuple specifying the number of rows and columns (num_rows, num_columns)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (26, 10))
    use_log_scale : bool, optional
        Whether to use logarithmic color scaling (default: False)
    show_colorbar : bool, optional
        Whether to show colorbars for each subplot (default: False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axs : numpy.ndarray
        Array of subplot axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define grid dimensions
    num_rows, num_columns = grid

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()
    
    # Set up colormap
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white')
    
    # Calculate scales for each group
    group_scales = {}
    for group in groups:
        relevant_images = [np.squeeze(image_data[event_index[0], :, :, i]) for i in group]
        if use_log_scale:
            vmin = 1e-10  # Small positive number for log scale
        else:
            vmin = 0
        vmax = max(np.max(img[img > 0]) if np.any(img > 0) else vmin for img in relevant_images)
        group_scales[tuple(group)] = (vmin, vmax)
    
    # Plot images
    for idx in range(len(labels)):
        img = np.squeeze(image_data[event_index[0], :, :, idx])
        masked_img = np.ma.masked_where(img <= 0, img)
        
        for group, (vmin, vmax) in group_scales.items():
            if idx in group:
                if use_log_scale:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                         norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                         vmin=vmin, vmax=vmax)
                break
        
        axs[idx].set_title(labels[idx], fontsize=20)
        
        # Add individual colorbar if requested
        if show_colorbar:
            divider = make_axes_locatable(axs[idx])
            cax = divider.append_axes("right", size="5%", pad=0.15)
            plt.colorbar(im, cax=cax)
        
        # Remove ticks
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
    
    plt.tight_layout()
    plt.show()



def alberto_image(pe_matrix, time_matrix, *maps):

    ch_z, ch_y = maps[0].shape  # Assuming all maps have the same shape
    n_eventos = pe_matrix.shape[0]
    
    # Initialize empty matrices for the maps
    pe_matrix_vis_map = np.zeros((n_eventos, ch_z, ch_y))
    pe_matrix_vuv_map = np.zeros((n_eventos, ch_z, ch_y))

    time_matrix_vis_map = np.zeros((n_eventos, ch_z, ch_y))
    time_matrix_vuv_map = np.zeros((n_eventos, ch_z, ch_y))

    # Fill the maps with the data
    
    
    for i in range(n_eventos):
        for j in range(ch_z):
            for k in range(ch_y):
                if maps[0][j][k] >= 0:
                    pe_matrix_vis_map[i][j][k] = pe_matrix[i][maps[0][j][k]]
                    time_matrix_vis_map[i][j][k] = time_matrix[i][maps[0][j][k]]
                if maps[1][j][k] >= 0:
                    pe_matrix_vuv_map[i][j][k] = pe_matrix[i][maps[1][j][k]]
                    time_matrix_vuv_map[i][j][k] = time_matrix[i][maps[1][j][k]]

    # Split and normalize the sensors of different radiation types into two layers
    pe_matrix_vis_map = np.hsplit(pe_matrix_vis_map, 2) / np.max(pe_matrix)
    pe_matrix_vuv_map = np.hsplit(pe_matrix_vuv_map, 2) / np.max(pe_matrix)

    time_matrix_vis_map = np.hsplit(time_matrix_vis_map, 2) / np.max(time_matrix)
    time_matrix_vuv_map = np.hsplit(time_matrix_vuv_map, 2) / np.max(time_matrix)

    # Create the image combining all the layers
    image = np.zeros((
        np.shape(pe_matrix_vis_map[0])[0],
        np.shape(pe_matrix_vis_map[0])[1],
        np.shape(pe_matrix_vis_map[0])[2],
        8
    ))

    image[:, :, :, 0] = pe_matrix_vis_map[0]
    image[:, :, :, 1] = pe_matrix_vis_map[1]
    image[:, :, :, 2] = pe_matrix_vuv_map[0]
    image[:, :, :, 3] = pe_matrix_vuv_map[1]

    image[:, :, :, 4] = time_matrix_vis_map[0]
    image[:, :, :, 5] = time_matrix_vis_map[1]
    image[:, :, :, 6] = time_matrix_vuv_map[0]
    image[:, :, :, 7] = time_matrix_vuv_map[1]

    return image
