import numpy as np
import matplotlib.pyplot as plt

def image_visualization(image, event_index, cell_size):
    """
    Plots event images for given indices.

    Parameters:
    - image_min: numpy array containing the images.
    - event_index: list or numpy array of indices to plot.
    - cell_width: int specifying the width of each cell in the plot.
    - cell_height: int specifying the height of each cell in the plot.
    """
    num_events = len(event_index)
    num_columns = 8  # Número fijo de columnas
    num_rows = num_events  # Número de filas igual al número de eventos

    # Calcular el tamaño de la figura basado en el número de filas y columnas
    figsize = (cell_size[1] * num_columns, cell_size[0] * num_rows)

    print("PE1: Visible/Volume -", "PE2: Visible/Volume +", 
    "PE3: Ultraviolet/Volume -", "PE4: Ultraviolet/Volume +", 
    "T1: Visible/Volume -", "T2: Visible/Volume +", 
    "T3: Ultraviolet/Volume -", "T4: Ultraviolet/Volume +")
    
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    
    for i in range(num_rows):
        for j in range(num_columns):
            axs[i, j].imshow(image[event_index[i], :, :, j], cmap='BuPu')
    
    plt.setp(axs, xticks=[], yticks=[])
    plt.show()

def plot_image_4comp(image_data, event_idx=570, figsize=(26, 10), use_log_scale=False, show_colorbar=False):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int, optional
        Index of the event to plot (default: 570)
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
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define constants
    num_columns = 4
    num_rows = 4
    labels = [
        "PE1: Coated PMT/Volume -", "PE2: Coated PMT/Volume +", "T1: Coated PMT/Volume -", "T2: Coated PMT/Volume +", 
        "PE1: Uncoated PMT/Volume -", "PE2: Uncoated PMT/Volume +", "T1: Uncoated PMT/Volume -", "T2: Uncoated PMT/Volume +", 
        "PE1: XA VIS/Volume -", "PE2: XA VIS/Volume +", "T1: XA VIS/Volume -", "T2: XA VIS/Volume +", 
        "PE1: XA VUV/Volume -", "PE2: XA VUV/Volume +", "T1: XA VUV/Volume -", "T2: XA VUV/Volume +"
    ]
    
    groups = [
        [0, 1, 4, 5],    # First group
        [2, 3, 6, 7],    # Second group
        [8, 9, 12, 13],  # Third group
        [10, 11, 14, 15] # Fourth group
    ]
    
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
    for idx in range(16):
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
    
    return fig, axs

# Example usage:
# Basic usage
# fig, axs = plot_image_4comp(image_min_4comp)

# With different event index
# fig, axs = plot_pmt_volumes(image_min_4comp, event_idx=100)

# With logarithmic scaling and colorbars
# fig, axs = plot_pmt_volumes(image_min_4comp, use_log_scale=True, show_colorbar=True)

# With custom figure size
# fig, axs = plot_pmt_volumes(image_min_4comp, figsize=(20, 8))

plt.show()