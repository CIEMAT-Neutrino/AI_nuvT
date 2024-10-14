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

