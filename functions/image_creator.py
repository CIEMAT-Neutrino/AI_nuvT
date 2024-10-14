import numpy as np

def assign_values(pe_matrix, time_matrix, map, pe_matrix_map, time_matrix_map):
    n_events, ch_z, ch_y = pe_matrix_map.shape
    for i in range(n_events):
        for j in range(ch_z):
            for k in range(ch_y):
                if map[j][k] >= 0:
                    if map[j][k] < pe_matrix.shape[1]:
                        pe_matrix_map[i][j][k] = pe_matrix[i][map[j][k]]
                        time_matrix_map[i][j][k] = time_matrix[i][map[j][k]]
                    else:
                        raise IndexError(f"IndexError: map[{j}][{k}] = {map[j][k]} is out of bounds for pe_matrix with shape {pe_matrix.shape}")

def image_creator(pe_matrix, time_matrix, vis_map, vuv_map):
    ch_z, ch_y = vis_map.shape
    n_events = pe_matrix.shape[0]

    # Inicializar matrices y asignarlas a la lista maps
    pe_matrix_vis_map, pe_matrix_vuv_map, time_matrix_vis_map, time_matrix_vuv_map = [np.zeros((n_events, ch_z, ch_y)) 
                                                                                        for _ in range(4)]

    # Asignar valores a las matrices
    assign_values(pe_matrix, time_matrix, vis_map, pe_matrix_vis_map, time_matrix_vis_map)
    assign_values(pe_matrix, time_matrix, vuv_map, pe_matrix_vuv_map, time_matrix_vuv_map)

    # Dividir y normalizar las matrices
    max_pe_matrix = np.max(pe_matrix)
    max_time_matrix = np.max(time_matrix)
    pe_matrix_vis_map, pe_matrix_vuv_map, time_matrix_vis_map, time_matrix_vuv_map = [
        np.hsplit(arr, 2) / max_val for arr, max_val in zip(
            [pe_matrix_vis_map, pe_matrix_vuv_map, time_matrix_vis_map, time_matrix_vuv_map],
            [max_pe_matrix, max_pe_matrix, max_time_matrix, max_time_matrix]
        )
    ]

    # Crear la imagen final
    image = np.zeros((n_events, ch_z, ch_y, 8))
    for i, map_pair in enumerate([pe_matrix_vis_map, pe_matrix_vuv_map, time_matrix_vis_map, time_matrix_vuv_map]):
        image[:, :, :, 2 * i] = map_pair[0]
        image[:, :, :, 2 * i + 1] = map_pair[1]

    return image