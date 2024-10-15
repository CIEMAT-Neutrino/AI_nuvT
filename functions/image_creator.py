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

    # Process each map
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]

    # Normalize matrices and split them
    pe_matrices_map = [np.hsplit(pe_mat, 2) / np.max(pe_matrix) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) / np.max(time_matrix) for time_mat in time_matrices_map]

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
