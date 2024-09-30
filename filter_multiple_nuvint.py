import awkward as ak

def filter_subarrays(*arrays):
    """
    Filter subarrays from the given lists of awkward arrays based on the condition
    that the subarray must not have exactly 2 elements.

    Parameters:
    - arrays: A variable number of lists of awkward arrays to evaluate for filtering.
      IMPORTANT -> nuvT must be the first variable

    Returns:
    - filtered_arrays: A list of filtered awkward arrays, where subarrays with exactly 2 elements are removed.
    """
    
    # Identify the indices of subarrays with exactly 2 elements from the first array
    indices_to_eliminate = [i for i, subarray in enumerate(arrays[0]) if ak.len(subarray) >= 2]
    
    # Eliminate the subarrays with exactly 2 elements from all input arrays
    filtered_arrays = [
        ak.Array([subarray for i, subarray in enumerate(array) if i not in indices_to_eliminate])
        for array in arrays
    ]
    
    return filtered_arrays

# Example of how to call the function:
# hit_nuvT_filtered, hit_PE_filtered, hit_ch_filtered, hit_t_filtered = filter_subarrays(hit_nuvT, hit_PE, hit_ch, hit_t)
