import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch


def image_creator_gen_onlype(pe_matrix, *maps):
    """
    Generalized function to create an image based on an arbitrary number of maps.
    
    :param pe_matrix: A 2D or 3D array of photoelectron counts.
    :param time_matrix: A 2D or 3D array of time values.
    :param maps: Any number of maps used for spatial distributions (vis_map, vuv_map, etc.).
    :return: An image array with channels corresponding to each map.
    """
    ch_y, ch_z = maps[0].shape  # Assuming all maps have the same shape
    n_events = pe_matrix.shape[0]

    # Create empty arrays to hold the results for each map
    map_count = len(maps)
    
    pe_matrices_map = [np.zeros((n_events, ch_y, ch_z)) for _ in range(map_count)]

    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]

        #NEW NORMALIZATION

        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])

        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])


    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    image = np.zeros((n_events, int(ch_y / 2), ch_z, 2 * map_count))


    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat in pe_matrices_map:
        image[:, :, :, channel] = pe_mat[0]
        image[:, :, :, channel + 1] = pe_mat[1]
        channel += 2

    return image



def image_creator_gen(pe_matrix, time_matrix, *maps):
    """
    Generalized function to create an image based on an arbitrary number of maps.
    
    :param pe_matrix: A 2D or 3D array of photoelectron counts.
    :param time_matrix: A 2D or 3D array of time values.
    :param maps: Any number of maps used for spatial distributions (vis_map, vuv_map, etc.).
    :return: An image array with channels corresponding to each map.
    """
    ch_y, ch_z = maps[0].shape  # Assuming all maps have the same shape
    n_events = pe_matrix.shape[0]

    # Create empty arrays to hold the results for each map
    map_count = len(maps)
    
    pe_matrices_map = [np.zeros((n_events, ch_y, ch_z)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_y, ch_z)) for _ in range(map_count)]

    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]

        #NEW NORMALIZATION

        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])
            time_matrices_map[idx] /= np.max(time_matrices_map[:2])
        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])
            time_matrices_map[idx] /= np.max(time_matrices_map[2:])

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    image = np.zeros((n_events, int(ch_y / 2), ch_z, 2 * map_count * 2))


    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        image[:, :, :, channel] = pe_mat[0]
        image[:, :, :, channel + 1] = pe_mat[1]
        image[:, :, :, channel + 2] = time_mat[0]
        image[:, :, :, channel + 3] = time_mat[1]
        channel += 4

    return image

def image_creator_gen_inv(pe_matrix, time_matrix, *maps):
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
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Get max time for this event
            max_time = np.max(time_matrices_map[idx][i])  
            
            # Get min nonzero time, but check if there are any nonzero values first
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0  # Avoid error
            
            # Apply transformation while ensuring zeros don't distort the range
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                max_time - time_matrices_map[idx][i] + min_time,  # Shift values
                0  # Ensure zeros remain zeros
            )
    
        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])
            time_matrices_map_inv[idx] /= np.max(time_matrices_map_inv[:2])
        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])
            time_matrices_map_inv[idx] /= np.max(time_matrices_map_inv[2:])


    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map_inv = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map_inv]

    # Create the final image with 2 channels per map (photoelectrons and time)
    image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count * 2))

    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map_inv):
        image[:, :, :, channel] = pe_mat[0]
        image[:, :, :, channel + 1] = pe_mat[1]
        image[:, :, :, channel + 2] = time_mat[0]
        image[:, :, :, channel + 3] = time_mat[1]
        channel += 4

    return image


import numpy as np

def image_creator_gen_inv_nuevo(pe_matrix, time_matrix, *maps):
    """
    Función generalizada para crear una imagen a partir de un número arbitrario de mapas.

    :param pe_matrix: Array 2D o 3D de cuentas de fotoelectrones.
    :param time_matrix: Array 2D o 3D de valores de tiempo.
    :param maps: Cualquier cantidad de mapas usados para distribuciones espaciales (vis_map, vuv_map, etc.).
    :return: Array de imagen con 2 canales por mapa (fotoelectrones y tiempo).
    """
    ch_z, ch_y = maps[0].shape  # Se asume que todos los mapas tienen la misma forma
    n_events = pe_matrix.shape[0]
    map_count = len(maps)

    # Crear arrays vacíos para almacenar los resultados por mapa
    pe_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Procesar cada mapa y normalizar de forma separada para dos grupos
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Procesar cada evento para el mapa actual
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Obtener el tiempo máximo para este evento
            max_time = np.max(time_matrices_map[idx][i])
            
            # Obtener el mínimo de los tiempos no nulos, si existen
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0
            
            # Aplicar transformación asegurando que los ceros se mantengan
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                max_time - time_matrices_map[idx][i] + min_time,  # Se desplazan los valores
                0
            )
    
        # Normalizar separadamente: los dos primeros mapas (primer grupo) y el resto (segundo grupo)
        if idx < 2:  # Primer grupo
            pe_matrices_map[idx] /= np.max([np.max(x) for x in pe_matrices_map[:2]])
            time_matrices_map_inv[idx] /= np.max([np.max(x) for x in time_matrices_map_inv[:2]])
        else:  # Segundo grupo
            pe_matrices_map[idx] /= np.max([np.max(x) for x in pe_matrices_map[2:]])
            time_matrices_map_inv[idx] /= np.max([np.max(x) for x in time_matrices_map_inv[2:]])

    # Dividir horizontalmente cada matriz en dos mitades
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map_inv = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map_inv]

    # Crear la imagen final: 2 canales por mapa (PE y tiempo)
    image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count))
    
    channel = 0
    for pe_halves, time_halves in zip(pe_matrices_map, time_matrices_map_inv):
        # Seleccionar la mitad con mayor cantidad de datos para los PE
        count_pe0 = np.count_nonzero(pe_halves[0])
        count_pe1 = np.count_nonzero(pe_halves[1])
        pe_selected = pe_halves[0] if count_pe0 >= count_pe1 else pe_halves[1]
        
        # Seleccionar la mitad con mayor cantidad de datos para el tiempo
        count_time0 = np.count_nonzero(time_halves[0])
        count_time1 = np.count_nonzero(time_halves[1])
        time_selected = time_halves[0] if count_time0 >= count_time1 else time_halves[1]
        
        # Asignar los datos seleccionados a la imagen final
        image[:, :, :, channel] = pe_selected
        image[:, :, :, channel + 1] = time_selected
        channel += 2

    return image


import numpy as np

def image_creator_gen_inv_nuevo2(pe_matrix, time_matrix, *maps):
    """
    Función generalizada para crear una imagen a partir de un número arbitrario de mapas.

    :param pe_matrix: Array 2D o 3D de cuentas de fotoelectrones.
    :param time_matrix: Array 2D o 3D de valores de tiempo.
    :param maps: Cualquier cantidad de mapas usados para distribuciones espaciales (vis_map, vuv_map, etc.).
    :return: Array de imagen con 2 canales por mapa (fotoelectrones y tiempo).
    """
    ch_z, ch_y = maps[0].shape  # Se asume que todos los mapas tienen la misma forma
    n_events = pe_matrix.shape[0]
    map_count = len(maps)

    # Crear arrays vacíos para almacenar los resultados por mapa
    pe_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Procesar cada mapa y normalizar de forma separada para dos grupos
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Procesar cada evento para el mapa actual
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Obtener el tiempo máximo para este evento
            max_time = np.max(time_matrices_map[idx][i])
            
            # Obtener el mínimo de los tiempos no nulos, si existen
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0
            
            # Aplicar transformación asegurando que los ceros se mantengan
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                max_time - time_matrices_map[idx][i] + min_time,  # Se desplazan los valores
                0
            )
    
        # Normalizar separadamente: los dos primeros mapas (primer grupo) y el resto (segundo grupo)
        if idx < 2:  # Primer grupo
            pe_max = np.max([np.max(x) for x in pe_matrices_map[:2]])
            time_max = np.max([np.max(x) for x in time_matrices_map_inv[:2]])
        else:  # Segundo grupo
            pe_max = np.max([np.max(x) for x in pe_matrices_map[2:]])
            time_max = np.max([np.max(x) for x in time_matrices_map_inv[2:]])
            
        pe_matrices_map[idx] = pe_matrices_map[idx] / pe_max if pe_max != 0 else pe_matrices_map[idx]
        time_matrices_map_inv[idx] = time_matrices_map_inv[idx] / time_max if time_max != 0 else time_matrices_map_inv[idx]

    # Dividir horizontalmente cada matriz en dos mitades
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map_inv = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map_inv]

    # Crear la imagen final: 2 canales por mapa (PE y tiempo)
    # NOTA: aquí cada "evento" tendrá asignada la mitad seleccionada de su mapa.
    image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count))
    
    # Ahora, para cada mapa, hacer la selección por evento
    for m, (pe_halves, time_halves) in enumerate(zip(pe_matrices_map, time_matrices_map_inv)):
        for i in range(n_events):
            # Para PE: seleccionar la mitad con mayor cantidad de datos para este evento
            count_pe0 = np.count_nonzero(pe_halves[0][i])
            count_pe1 = np.count_nonzero(pe_halves[1][i])
            pe_selected = pe_halves[0][i] if count_pe0 >= count_pe1 else pe_halves[1][i]
            
            # Para tiempo: seleccionar la mitad con mayor cantidad de datos para este evento
            count_time0 = np.count_nonzero(time_halves[0][i])
            count_time1 = np.count_nonzero(time_halves[1][i])
            time_selected = time_halves[0][i] if count_time0 >= count_time1 else time_halves[1][i]
            
            # Asignar los datos seleccionados a la imagen final
            image[i, :, :, 2 * m] = pe_selected
            image[i, :, :, 2 * m + 1] = time_selected

    return image




def image_creator_gen_2images(pe_matrix, time_matrix, *maps):
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
       # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]

        #NEW NORMALIZATION

        # Normalize the first two maps (first eight channels) separately from the last two
        if idx < 2:  # First group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[:2])
            time_matrices_map[idx] /= np.max(time_matrices_map[:2])
        else:  # Second group
            pe_matrices_map[idx] /= np.max(pe_matrices_map[2:])
            time_matrices_map[idx] /= np.max(time_matrices_map[2:])

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    pe_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))
    time_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))  
    
    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        pe_image[:, :, :, channel] = pe_mat[0]
        pe_image[:, :, :, channel + 1] = pe_mat[1]
        time_image[:, :, :, channel] = time_mat[0]
        time_image[:, :, :, channel + 1] = time_mat[1]
        channel += 2

    return pe_image, time_image


def image_creator_gen_2images_inv(pe_matrix, time_matrix, *maps):
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
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Get max time for this event
            max_time = np.max(time_matrices_map[idx][i])  
            
            # Get min nonzero time, but check if there are any nonzero values first
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0  # Avoid error
            
            # Apply transformation while ensuring zeros don't distort the range
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                max_time - time_matrices_map[idx][i] + min_time,  # Shift values
                0  # Ensure zeros remain zeros
            )
    
        pe_matrices_map[idx] /= np.max(pe_matrices_map[idx])  # Normalizar cada mapa de PE individualmente
    
    # Compute the global max across all time matrices
    global_max_time = max(np.max(time_mat) for time_mat in time_matrices_map_inv)

    # Normalize using the global max
    for idx in range(len(time_matrices_map_inv)):
        time_matrices_map_inv[idx] /= global_max_time

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    pe_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))
    time_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2* map_count))  

    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        pe_image[:, :, :, channel] = pe_mat[0]
        pe_image[:, :, :, channel + 1] = pe_mat[1]
        time_image[:, :, :, channel] = time_mat[0]
        time_image[:, :, :, channel + 1] = time_mat[1]
        channel += 2

    return pe_image, time_image

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def image_creator_gen_2images_inv_v2502_prueba(pe_matrix, time_matrix, *maps):
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
    time_matrices_map_inv = [np.zeros((n_events, ch_z, ch_y)) for _ in range(map_count)]
    
    # Process each map and normalize separately for two groups
    for idx, map_ in enumerate(maps):
        valid_map = (map_ >= 0) & (map_ < pe_matrix.shape[1])
        
        # Process each event for the current map
        for i in range(n_events):
            pe_matrices_map[idx][i][valid_map] = pe_matrix[i][map_[valid_map]]
            time_matrices_map[idx][i][valid_map] = time_matrix[i][map_[valid_map]]
            
            # Get max time for this event
            max_time = np.max(time_matrices_map[idx][i])  
            
            # Get min nonzero time, but check if there are any nonzero values first
            nonzero_times = time_matrices_map[idx][i][time_matrices_map[idx][i] > 0]
            min_time = np.min(nonzero_times) if nonzero_times.size > 0 else 0  # Avoid error
            
            # Apply transformation while ensuring zeros don't distort the range, and set negative values to zero
            time_matrices_map_inv[idx][i] = np.where(
                time_matrices_map[idx][i] != 0,
                np.maximum(max_time - time_matrices_map[idx][i] + min_time, 0),  # Set negative values to 0
                0  # Ensure zeros remain zeros
            )

    # Check for NaN or infinite values in inputs
    if np.any(np.isnan(pe_matrix)) or np.any(np.isnan(time_matrix)) or np.any(np.isinf(pe_matrix)) or np.any(np.isinf(time_matrix)):
        raise ValueError("NaN or infinite values detected in input matrices")

    # Normalización global de los mapas de fotoelectrones using MinMaxScaler
    pe_all_values = np.concatenate([pe_mat.flatten() for pe_mat in pe_matrices_map], axis=0)
    scaler_pe = MinMaxScaler(feature_range=(0, 1))
    scaled_pe = scaler_pe.fit_transform(pe_all_values.reshape(-1, 1)).flatten()
    
    # Reshape and assign back to pe_matrices_map
    offset = 0
    for idx in range(len(pe_matrices_map)):
        size = pe_matrices_map[idx].size
        pe_matrices_map[idx] = scaled_pe[offset:offset + size].reshape(pe_matrices_map[idx].shape)
        offset += size

    # Normalización global de las matrices de tiempo using MinMaxScaler
    time_all_values = np.concatenate([time_mat.flatten() for time_mat in time_matrices_map_inv], axis=0)

    # Check for NaN or infinite values in time_matrices_map_inv
    if np.any(np.isnan(time_all_values)) or np.any(np.isinf(time_all_values)):
        raise ValueError("NaN or infinite values detected in time_matrices_map_inv before normalization")

    # Filter out negative values and zeros to find the minimum positive value
    positive_nonzero_values = time_all_values[time_all_values > 0]
    if len(positive_nonzero_values) == 0:
        raise ValueError("No positive, non-zero values found in time_matrices_map_inv")

    # Use MinMaxScaler to normalize time values to [0, 1], preserving zeros
    scaler_time = MinMaxScaler(feature_range=(0, 1))
    scaled_time = np.zeros_like(time_all_values)  # Initialize with zeros
    if len(positive_nonzero_values) > 0:
        # Scale only non-zero values
        mask = time_all_values > 0
        try:
            scaled_time[mask] = scaler_time.fit_transform(time_all_values[mask].reshape(-1, 1)).flatten()
        except Exception as e:
            raise ValueError(f"Error in MinMaxScaler normalization: {str(e)}")
    
    # Reshape and assign back to time_matrices_map_inv, ensuring [0, 1]
    offset = 0
    for idx in range(len(time_matrices_map_inv)):
        size = time_matrices_map_inv[idx].size
        time_matrices_map_inv[idx] = scaled_time[offset:offset + size].reshape(time_matrices_map_inv[idx].shape)
        # Explicitly clip to [0, 1] as a final safeguard
        time_matrices_map_inv[idx] = np.clip(time_matrices_map_inv[idx], 0, 1)
        offset += size

    # Verify the output is within [0, 1] for both pe and time matrices
    if np.any(np.concatenate(pe_matrices_map) < 0) or np.any(np.concatenate(pe_matrices_map) > 1):
        raise ValueError("Normalized photoelectron values are outside the [0, 1] range")
    if np.any(np.concatenate(time_matrices_map_inv) < 0) or np.any(np.concatenate(time_matrices_map_inv) > 1):
        raise ValueError("Normalized time values are outside the [0, 1] range")

    # Split and scale matrices for each map
    pe_matrices_map = [np.hsplit(pe_mat, 2) for pe_mat in pe_matrices_map]
    time_matrices_map = [np.hsplit(time_mat, 2) for time_mat in time_matrices_map]

    # Create the final image with 2 channels per map (photoelectrons and time)
    pe_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count))
    time_image = np.zeros((n_events, int(ch_z / 2), ch_y, 2 * map_count))  

    # Populate the image's channels with the data from each map
    channel = 0
    for pe_mat, time_mat in zip(pe_matrices_map, time_matrices_map):
        pe_image[:, :, :, channel] = pe_mat[0]
        pe_image[:, :, :, channel + 1] = pe_mat[1]
        time_image[:, :, :, channel] = time_mat[0]
        time_image[:, :, :, channel + 1] = time_mat[1]
        channel += 2

    return pe_image, time_image

def select_non_empty_half(matrix, method="max"):
    """
    Selects the half of the matrix (left or right) that contains more meaningful data.

    The selection is based on a specified method:
    - "max": Chooses the half with the highest maximum value.
    - "sum": Chooses the half with the highest total sum of values.
    - "nonzero": Chooses the half with the most nonzero elements.
    - "mean_top": Chooses the half with the highest mean of the top N values.

    Parameters:
    -----------
    matrix : np.ndarray
        The input 2D or 3D matrix (e.g., (n_events, height, width)).
    method : str, optional
        The selection criterion ("max", "sum", "nonzero", "mean_top"), by default "max".

    Returns:
    --------
    np.ndarray
        The selected half of the matrix.
    """

    # Ensure the matrix has an even number of columns (width)
    if matrix.shape[1] % 2 != 0:
        raise ValueError("The matrix width must be evenly divisible by 2 to split it equally.")

    # Split matrix into left and right halves
    left_half, right_half = np.hsplit(matrix, 2)

    if method == "max":
        left_score = np.max(left_half)
        right_score = np.max(right_half)
    elif method == "sum":
        left_score = np.sum(left_half)
        right_score = np.sum(right_half)
    elif method == "nonzero":
        left_score = np.count_nonzero(left_half)
        right_score = np.count_nonzero(right_half)
    elif method == "mean_top":
        top_n = 5  # Number of top values to consider
        left_score = np.mean(np.sort(left_half.flatten())[-top_n:])
        right_score = np.mean(np.sort(right_half.flatten())[-top_n:])
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from 'max', 'sum', 'nonzero', or 'mean_top'.")

    return left_half if left_score >= right_score else right_half


def plot_image(image_data, event_idx, labels, groups, grid, figsize=(26, 10), use_log_scale=False, show_colorbar=False):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int
        Index of the event to plot
    labels : list of str
        Labels for each subplot
    groups : list of list of int
        Grouping of the image indices for shared scaling
    grid : tuple
        Tuple specifying the number of rows and columns (num_rows, num_columns)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (26, 10))
    use_log_scale : bool, optional
        Whether to use logarithmic color scaling (default: False)
    show_colorbar : bool, optional
        Whether to show colorbars for each subplot (default: False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axs : numpy.ndarray
        Array of subplot axes
    """
    
    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define grid dimensions
    num_rows, num_columns = grid

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()
    
    # Set up colormap
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white')
    
    # Calculate scales for each group
    group_scales = {}
    for group in groups:
        relevant_images = [np.squeeze(image_data[event_index[0], :, :, i]) for i in group]
        if use_log_scale:
            vmin = 1e-10  # Small positive number for log scale
        else:
            vmin = 0
        vmax = max(np.max(img[img > 0]) if np.any(img > 0) else vmin for img in relevant_images)
        group_scales[tuple(group)] = (vmin, vmax)
    
    # Plot images
    for idx in range(len(labels)):
        img = np.squeeze(image_data[event_index[0], :, :, idx])
        masked_img = np.ma.masked_where(img <= 0, img)
        
        for group, (vmin, vmax) in group_scales.items():
            if idx in group:
                if use_log_scale:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       vmin=vmin, vmax=vmax)
                break
        
        axs[idx].set_title(labels[idx], fontsize=20)
        
        # Add colorbar if requested
        if show_colorbar:
            divider = make_axes_locatable(axs[idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=20)

            
    # Remove ticks
    plt.setp(axs, xticks=[], yticks=[])
    
    plt.tight_layout()
    
    plt.show()

def plot_image2(image_data, event_idx, labels, groups, grid, figsize=(26, 10), use_log_scale=False, show_colorbar=True):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int
        Index of the event to plot
    labels : list of str
        Labels for each subplot
    groups : list of list of int
        Grouping of the image indices for shared scaling
    grid : tuple
        Tuple specifying the number of rows and columns (num_rows, num_columns)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (26, 10))
    use_log_scale : bool, optional
        Whether to use logarithmic color scaling (default: False)
    show_colorbar : bool, optional
        Whether to show colorbars for each subplot (default: False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axs : numpy.ndarray
        Array of subplot axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define grid dimensions
    num_rows, num_columns = grid

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()
    
    # Set up colormap
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white')
    
    # Calculate scales for each group
    group_scales = {}
    for group in groups:
        relevant_images = [np.squeeze(image_data[event_index[0], :, :, i]) for i in group]
        if use_log_scale:
            vmin = 1e-10  # Small positive number for log scale
        else:
            vmin = 0
        vmax = max(np.max(img[img > 0]) if np.any(img > 0) else vmin for img in relevant_images)
        group_scales[tuple(group)] = (vmin, vmax)
    
    # Plot images
    for idx in range(len(labels)):
        img = np.squeeze(image_data[event_index[0], :, :, idx])
        masked_img = np.ma.masked_where(img <= 0, img)
        
        for group, (vmin, vmax) in group_scales.items():
            if idx in group:
                if use_log_scale:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                         norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                         vmin=vmin, vmax=vmax)
                break
        
        axs[idx].set_title(labels[idx], fontsize=20)
        
        # Add individual colorbar if requested
        if show_colorbar:
            divider = make_axes_locatable(axs[idx])
            cax = divider.append_axes("right", size="5%", pad=0.15)
            plt.colorbar(im, cax=cax)
        
        # Remove ticks
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
    
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_image_axis(image_data, event_idx, labels, groups, grid, figsize=(26, 10), use_log_scale=False, show_colorbar=False):
    """
    Plot created images with grouped color scaling.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        4D array containing the image data
    event_idx : int
        Index of the event to plot
    labels : list of str
        Labels for each subplot
    groups : list of list of int
        Grouping of the image indices for shared scaling
    grid : tuple
        Tuple specifying the number of rows and columns (num_rows, num_columns)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (26, 10))
    use_log_scale : bool, optional
        Whether to use logarithmic color scaling (default: False)
    show_colorbar : bool, optional
        Whether to show colorbars for each subplot (default: False)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axs : numpy.ndarray
        Array of subplot axes
    """
    
    # Convert event index to array
    event_index = np.array([event_idx])
    
    # Define grid dimensions
    num_rows, num_columns = grid

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_columns, figsize=figsize)
    axs = axs.flatten()
    
    # Set up colormap
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white')
    
    # Calculate scales for each group
    group_scales = {}
    for group in groups:
        relevant_images = [np.squeeze(image_data[event_index[0], :, :, i]) for i in group]
        if use_log_scale:
            vmin = 1e-10  # Small positive number for log scale
        else:
            vmin = 0
        vmax = max(np.max(img[img > 0]) if np.any(img > 0) else vmin for img in relevant_images)
        group_scales[tuple(group)] = (vmin, vmax)
    
    # Plot images
    for idx in range(len(labels)):
        img = np.squeeze(image_data[event_index[0], :, :, idx])
        masked_img = np.ma.masked_where(img <= 0, img)
        
        for group, (vmin, vmax) in group_scales.items():
            if idx in group:
                if use_log_scale:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs[idx].imshow(masked_img, cmap=cmap, 
                                       vmin=vmin, vmax=vmax)
                break
        
        axs[idx].set_title(labels[idx], fontsize=20)
        
        # Add colorbar if requested
        if show_colorbar:
            divider = make_axes_locatable(axs[idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=20)
        
        # Add arrows indicating the y and z axes
        # Eje Y (apunta hacia arriba en el gráfico)
        axs[idx].annotate('', xy=(0.1, 0.1), xytext=(0.1, 0.9), 
                          arrowprops=dict(facecolor='red', shrink=0.05, width=2))
        
        # Eje Z (apunta hacia la derecha)
        axs[idx].annotate('', xy=(0.1, 0.1), xytext=(0.9, 0.1), 
                          arrowprops=dict(facecolor='blue', shrink=0.05, width=2))

        # Etiquetas de los ejes
        axs[idx].text(0.1, 0.95, 'y', color='red', fontsize=14)
        axs[idx].text(0.9, 0.05, 'z', color='blue', fontsize=14)

    # Remove ticks
    plt.setp(axs, xticks=[], yticks=[])
    
    plt.tight_layout()
    
    plt.show()
