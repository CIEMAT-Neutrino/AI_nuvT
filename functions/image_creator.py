import numpy as np

def assign_values(fotoelectrones, tiempos, map, fotoelectrones_map, tiempos_map):
    n_eventos, ch_z, ch_y = fotoelectrones_map.shape
    for i in range(n_eventos):
        for j in range(ch_z):
            for k in range(ch_y):
                if map[j][k] >= 0:
                    if map[j][k] < fotoelectrones.shape[1]:
                        fotoelectrones_map[i][j][k] = fotoelectrones[i][map[j][k]]
                        tiempos_map[i][j][k] = tiempos[i][map[j][k]]
                    else:
                        raise IndexError(f"IndexError: map[{j}][{k}] = {map[j][k]} is out of bounds for fotoelectrones with shape {fotoelectrones.shape}")

def image_creator(fotoelectrones, tiempos, vis_map, vuv_map):
    ch_z, ch_y = vis_map.shape
    n_eventos = fotoelectrones.shape[0]

    # Inicializar matrices y asignarlas a la lista maps
    fotoelectrones_vis_map, fotoelectrones_vuv_map, tiempos_vis_map, tiempos_vuv_map = [np.zeros((n_eventos, ch_z, ch_y)) 
                                                                                        for _ in range(4)]

    # Asignar valores a las matrices
    assign_values(fotoelectrones, tiempos, vis_map, fotoelectrones_vis_map, tiempos_vis_map)
    assign_values(fotoelectrones, tiempos, vuv_map, fotoelectrones_vuv_map, tiempos_vuv_map)

    # Dividir y normalizar las matrices
    max_fotoelectrones = np.max(fotoelectrones)
    max_tiempos = np.max(tiempos)
    fotoelectrones_vis_map, fotoelectrones_vuv_map, tiempos_vis_map, tiempos_vuv_map = [
        np.hsplit(arr, 2) / max_val for arr, max_val in zip(
            [fotoelectrones_vis_map, fotoelectrones_vuv_map, tiempos_vis_map, tiempos_vuv_map],
            [max_fotoelectrones, max_fotoelectrones, max_tiempos, max_tiempos]
        )
    ]

    # Crear la imagen final
    image = np.zeros((n_eventos, ch_z, ch_y, 8))
    for i, map_pair in enumerate([fotoelectrones_vis_map, fotoelectrones_vuv_map, tiempos_vis_map, tiempos_vuv_map]):
        image[:, :, :, 2 * i] = map_pair[0]
        image[:, :, :, 2 * i + 1] = map_pair[1]

    return image