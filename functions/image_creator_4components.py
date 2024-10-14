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

def image_creator3(pe_matrix, time_matrix, coated_pmt_map, uncoated_pmt_map, xarap_vis_map, xarap_vuv_map):
    ch_z, ch_y = coated_pmt_map.shape
    n_events = pe_matrix.shape[0]

    # Inicializar matrices y asignarlas a la lista maps
    pe_matrix_coated_map, pe_matrix_uncoated_map, pe_matrix_xarap_vis_map, pe_matrix_xarap_vuv_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(4)]
    time_matrix_coated_map, time_matrix_uncoated_map, time_matrix_xarap_vis_map, time_matrix_xarap_vuv_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(4)]

    # Asignar valores a las matrices usando las cuatro mapas
    assign_values(pe_matrix, time_matrix, coated_pmt_map, pe_matrix_coated_map, time_matrix_coated_map)
    assign_values(pe_matrix, time_matrix, uncoated_pmt_map, pe_matrix_uncoated_map, time_matrix_uncoated_map)
    assign_values(pe_matrix, time_matrix, xarap_vis_map, pe_matrix_xarap_vis_map, time_matrix_xarap_vis_map)
    assign_values(pe_matrix, time_matrix, xarap_vuv_map, pe_matrix_xarap_vuv_map, time_matrix_xarap_vuv_map)

    # Dividir y normalizar las matrices
    max_pe_matrix = np.max(pe_matrix)
    max_time_matrix = np.max(time_matrix)
    pe_matrix_coated_map, pe_matrix_uncoated_map, pe_matrix_xarap_vis_map, pe_matrix_xarap_vuv_map = [
        np.hsplit(arr, 2) / max_pe_matrix for arr in [
            pe_matrix_coated_map, pe_matrix_uncoated_map, pe_matrix_xarap_vis_map, pe_matrix_xarap_vuv_map
        ]
    ]
    time_matrix_coated_map, time_matrix_uncoated_map, time_matrix_xarap_vis_map, time_matrix_xarap_vuv_map = [
        np.hsplit(arr, 2) / max_time_matrix for arr in [
            time_matrix_coated_map, time_matrix_uncoated_map, time_matrix_xarap_vis_map, time_matrix_xarap_vuv_map
        ]
    ]

    # Crear la imagen final con 16 canales (8 mapas * 2 cada uno: pe_matrix y time_matrix)
    image = np.zeros((int(np.shape(pe_matrix_coated_map[0])[0]), 
                  int(np.shape(pe_matrix_coated_map[0])[1]), 
                  int(np.shape(pe_matrix_coated_map[0])[2]), 16))

    image[:, :, :, 0] = pe_matrix_coated_map[0]
    image[:, :, :, 1] = pe_matrix_coated_map[1]
    image[:, :, :, 2] = pe_matrix_uncoated_map[0]
    image[:, :, :, 3] = pe_matrix_uncoated_map[1]
    image[:, :, :, 4] = pe_matrix_xarap_vis_map[0]
    image[:, :, :, 5] = pe_matrix_xarap_vis_map[1]
    image[:, :, :, 6] = pe_matrix_xarap_vuv_map[0]
    image[:, :, :, 7] = pe_matrix_xarap_vuv_map[1]

    image[:, :, :, 8] = time_matrix_coated_map[0]
    image[:, :, :, 9] = time_matrix_coated_map[1]
    image[:, :, :, 10] = time_matrix_uncoated_map[0]
    image[:, :, :, 11] = time_matrix_uncoated_map[1]
    image[:, :, :, 12] = time_matrix_xarap_vis_map[0]
    image[:, :, :, 13] = time_matrix_xarap_vis_map[1]
    image[:, :, :, 14] = time_matrix_xarap_vuv_map[0]
    image[:, :, :, 15] = time_matrix_xarap_vuv_map[1]


    return image

