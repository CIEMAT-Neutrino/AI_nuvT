import numpy as np

def image_creator2(pe_matrix, time_matrix, vis_map, vuv_map):
 
  # Creamos la matriz con distribución espacial
  ch_z, ch_y = vis_map.shape
  n_events = pe_matrix.shape[0]
    
  pe_matrix_vis_map, pe_matrix_vuv_map, time_matrix_vis_map, time_matrix_vuv_map = (np.zeros((n_events, ch_z, ch_y)) for _ in range(4))

  for i in range(n_events):
    for j in range(ch_z):
      for k in range(ch_y):
        if vis_map[j][k] >= 0:
          if vis_map[j][k] < pe_matrix.shape[1]:
            pe_matrix_vis_map[i][j][k] = pe_matrix[i][vis_map[j][k]]
            time_matrix_vis_map[i][j][k] = time_matrix[i][vis_map[j][k]]
          else:
            print(f"IndexError: vis_map[{j}][{k}] = {vis_map[j][k]} is out of bounds for pe_matrix with shape {pe_matrix.shape}")
        if vuv_map[j][k] >= 0:
          if vuv_map[j][k] < pe_matrix.shape[1]:
            pe_matrix_vuv_map[i][j][k] = pe_matrix[i][vuv_map[j][k]]
            time_matrix_vuv_map[i][j][k] = time_matrix[i][vuv_map[j][k]]
          else:
            print(f"IndexError: vuv_map[{j}][{k}] = {vuv_map[j][k]} is out of bounds for pe_matrix with shape {pe_matrix.shape}")

  # Dividimos los sensores de diferente radiación en dos capas distintas y también por volúmenes

  
  pe_matrix_vis_map = np.hsplit(pe_matrix_vis_map, 2) / np.max(pe_matrix)
  pe_matrix_vuv_map = np.hsplit(pe_matrix_vuv_map, 2) / np.max(pe_matrix)

  time_matrix_vis_map = np.hsplit(time_matrix_vis_map, 2) / np.max(time_matrix)
  time_matrix_vuv_map = np.hsplit(time_matrix_vuv_map, 2) / np.max(time_matrix)

  image = np.zeros((np.shape(pe_matrix_vis_map[0])[0], np.shape(pe_matrix_vis_map[0])[1], np.shape(pe_matrix_vis_map[0])[2], 8))

  image[:, :, :, 0] = pe_matrix_vis_map[0]
  image[:, :, :, 1] = pe_matrix_vis_map[1]
  image[:, :, :, 2] = pe_matrix_vuv_map[0]
  image[:, :, :, 3] = pe_matrix_vuv_map[1]

  image[:, :, :, 4] = time_matrix_vis_map[0]
  image[:, :, :, 5] = time_matrix_vis_map[1]
  image[:, :, :, 6] = time_matrix_vuv_map[0]
  image[:, :, :, 7] = time_matrix_vuv_map[1]

  return image