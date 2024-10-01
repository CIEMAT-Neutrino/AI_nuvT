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

    n_eventos = len(hit_ch)

    # Create matrices to store the number of photoelectrons and times for each event and channel
    fotoelectrones = np.zeros((n_eventos, n_canales))
    tiempos = np.zeros((n_eventos, n_canales))

    # Populate the matrices
    for i in range(n_eventos):
        for j in range(len(hit_PE[i])):
            for k, l, t in zip(hit_PE[i][j], hit_ch[i][j], hit_t[i][j]):
                fotoelectrones[i][l] += k
                tiempos[i][l] += k * t  # Weight times by charge

    # Normalize tiempos using the sum of weights
    for i in range(n_eventos):
        for j in range(n_canales):
            if fotoelectrones[i][j] != 0:
                tiempos[i][j] /= fotoelectrones[i][j]
            else:
                tiempos[i][j] = 0

    return fotoelectrones, tiempos

# Example usage
# fotoelectrones, tiempos = process_photoelectrons(hit_PE, hit_ch, hit_t)
