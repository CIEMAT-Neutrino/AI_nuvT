import matplotlib.pyplot as plt
import awkward as ak
import numpy as np


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


def sample_awkward_arrays(awkward_arrays, sample_size=100000, chunk_size=10000):
    # Step 1: Determine total number of elements across all awkward arrays
    total_elements = sum(len(ak.flatten(array)) for array in awkward_arrays)
    
    # Step 2: Randomly select 'sample_size' indices from the total number of elements
    selected_indices = np.random.choice(total_elements, size=sample_size, replace=False)
    selected_indices.sort()  # Sort to make checking within chunks easier

    sampled_data = []
    current_index = 0  # Track position in the flattened awkward arrays

    # Step 3: Process arrays in chunks
    for array in awkward_arrays:
        flattened_array = ak.flatten(array)
        num_elements = len(flattened_array)
        
        start = 0
        while start < num_elements:
            end = min(start + chunk_size, num_elements)
            chunk = flattened_array[start:end]
            numpy_chunk = ak.to_numpy(chunk)

            # Step 4: Check if any of the selected indices fall within this chunk
            chunk_indices = np.where((selected_indices >= current_index) & (selected_indices < current_index + len(numpy_chunk)))[0]
            if len(chunk_indices) > 0:
                # Calculate local indices within the chunk
                local_indices = selected_indices[chunk_indices] - current_index
                sampled_data.extend(numpy_chunk[local_indices])

            # Update index tracking
            current_index += len(numpy_chunk)
            start = end

            # Step 5: Stop if we've reached the desired sample size
            if len(sampled_data) >= sample_size:
                break
        
        if len(sampled_data) >= sample_size:
            break
    
    return np.array(sampled_data[:sample_size])  # Ensure we return exactly the sample size



def plot_variable_histograms(hit_nuvT_filtered, hit_t_filtered, hit_ch_filtered):
    # Flatten the data
    flattened_hit_nuvT_filtered = ak.flatten(ak.Array(hit_nuvT_filtered))
    flattened_hit_t_filtered = sample_awkward_arrays(hit_t_filtered)
    flattened_hit_ch_filtered = sample_awkward_arrays(hit_ch_filtered)

    # Create subplots with 3 rows and 1 column
    fig, axs = plt.subplots(3, 1, figsize=(20, 25))
    fig.subplots_adjust(hspace=0.4)  # Adjust spacing between subplots

    # First histogram: nuvT
    axs[0].hist(flattened_hit_nuvT_filtered, bins=300, edgecolor='black', alpha=0.7)
    axs[0].set_xlabel('Time [ns]', fontsize=14)
    axs[0].set_ylabel('# OpHits', fontsize=14)
    axs[0].set_title('NuvT Histogram', fontsize=16, fontweight='bold')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

    # Second histogram: ophit_time
    axs[1].hist(flattened_hit_t_filtered, bins=300, edgecolor='black', alpha=0.7)
    axs[1].set_xlabel(r'Time [$\mu$s]', fontsize=14)
    axs[1].set_ylabel('# OpHits', fontsize=14)
    axs[1].set_title('Ophit Time Histogram', fontsize=16, fontweight='bold')
    axs[1].set_xlim(0, 15)  # Set x-axis limits
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Third histogram: channel distribution
    axs[2].hist(flattened_hit_ch_filtered, bins=range(0, 312), edgecolor='black', alpha=0.7)
    axs[2].set_xlabel('# Channel', fontsize=14)
    axs[2].set_ylabel('# OpHits', fontsize=14)
    axs[2].set_title('Channel Distribution Histogram', fontsize=16, fontweight='bold')
    axs[2].set_xticks(range(0, 312, 10))
    axs[2].set_xlim(0, 312)  # Set x-axis limits
    axs[2].set_ylim(0, 1000)  # Set y-axis limits
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()




