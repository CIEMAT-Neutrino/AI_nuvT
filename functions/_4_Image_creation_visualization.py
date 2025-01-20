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
