import matplotlib.pyplot as plt

def plot_flash_time_distribution(flattened_data, bins=100, xlim=None, xticks=None):
    """
    Plots a histogram for flash time distribution.

    Parameters:
    - flattened_data: Array-like, the data to plot.
    - bins: Number of histogram bins (default 100).
    - x_label: Label for the x-axis (default r'time [$\mu$s]').
    - y_label: Label for the y-axis (default 'nº flashes').
    - title: Title for the plot (default 'Flash time distribution').
    - color: Color of the histogram (default 'orange').
    - xlim: Tuple of (min, max) for x-axis limits (optional).
    - xticks: List or range for custom x-axis ticks (optional).
    """
    
    plt.figure(figsize=(15, 6))
    plt.hist(flattened_data, bins=bins, color='orange')
    
    plt.xlabel(r'time [$\mu$s]')
    plt.ylabel('nº flashes')
    plt.title('Flash time distribution')
    
    if xlim:
        plt.xlim(xlim)
    
    if xticks:
        plt.xticks(xticks)
    
    plt.show()
