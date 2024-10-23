import numpy as np

def process_photoelectrons(hit_PE, hit_ch, hit_t, n_canales=312):
    """
    Process the filtered hit data to compute the number of photoelectrons and weighted times.

    Parameters:
    hit_PE (list of awkward arrays): Filtered hit photoelectron data.
    hit_ch (list of awkward arrays): Filtered channel data.
    hit_t (list of awkward arrays): Filtered time data.
    n_canales (int): Number of channels. Default is 312.

    Returns:
    tuple: Two numpy arrays containing the number of photoelectrons and the weighted times.
    """

    n_events = len(hit_ch)

    # Create matrices to store the number of photoelectrons and times for each event and channel
    pe_matrix = np.zeros((n_events, n_canales))
    time_matrix = np.zeros((n_events, n_canales))

    # Populate the matrices
    for i in range(n_events):
        for j in range(len(hit_PE[i])):
            for k, l, t in zip(hit_PE[i][j], hit_ch[i][j], hit_t[i][j]):
                pe_matrix[i][l] += k
                time_matrix[i][l] += k * t  # Weight times by charge

    # Normalize time_matrix using the sum of weights
    for i in range(n_events):
        for j in range(n_canales):
            if pe_matrix[i][j] != 0:
                time_matrix[i][j] /= pe_matrix[i][j]
            else:
                time_matrix[i][j] = 0

    return pe_matrix, time_matrix

# Example usage
# pe_matrix, time_matrix = process_photoelectrons(hit_PE, hit_ch, hit_t)
