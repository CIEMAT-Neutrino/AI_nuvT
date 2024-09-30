import numpy as np

def image_creator2(fotoelectrones, tiempos, vis_map, vuv_map):
 
  # Creamos la matriz con distribución espacial
  ch_z, ch_y = vis_map.shape
  n_eventos = fotoelectrones.shape[0]
    
  fotoelectrones_vis_map, fotoelectrones_vuv_map, tiempos_vis_map, tiempos_vuv_map = (np.zeros((n_eventos, ch_z, ch_y)) for _ in range(4))

  for i in range(n_eventos):
    for j in range(ch_z):
      for k in range(ch_y):
        if vis_map[j][k] >= 0:
          if vis_map[j][k] < fotoelectrones.shape[1]:
            fotoelectrones_vis_map[i][j][k] = fotoelectrones[i][vis_map[j][k]]
            tiempos_vis_map[i][j][k] = tiempos[i][vis_map[j][k]]
          else:
            print(f"IndexError: vis_map[{j}][{k}] = {vis_map[j][k]} is out of bounds for fotoelectrones with shape {fotoelectrones.shape}")
        if vuv_map[j][k] >= 0:
          if vuv_map[j][k] < fotoelectrones.shape[1]:
            fotoelectrones_vuv_map[i][j][k] = fotoelectrones[i][vuv_map[j][k]]
            tiempos_vuv_map[i][j][k] = tiempos[i][vuv_map[j][k]]
          else:
            print(f"IndexError: vuv_map[{j}][{k}] = {vuv_map[j][k]} is out of bounds for fotoelectrones with shape {fotoelectrones.shape}")

  # Dividimos los sensores de diferente radiación en dos capas distintas y también por volúmenes

  
  fotoelectrones_vis_map = np.hsplit(fotoelectrones_vis_map, 2) / np.max(fotoelectrones)
  fotoelectrones_vuv_map = np.hsplit(fotoelectrones_vuv_map, 2) / np.max(fotoelectrones)

  tiempos_vis_map = np.hsplit(tiempos_vis_map, 2) / np.max(tiempos)
  tiempos_vuv_map = np.hsplit(tiempos_vuv_map, 2) / np.max(tiempos)

  image = np.zeros((np.shape(fotoelectrones_vis_map[0])[0], np.shape(fotoelectrones_vis_map[0])[1], np.shape(fotoelectrones_vis_map[0])[2], 8))

  image[:, :, :, 0] = fotoelectrones_vis_map[0]
  image[:, :, :, 1] = fotoelectrones_vis_map[1]
  image[:, :, :, 2] = fotoelectrones_vuv_map[0]
  image[:, :, :, 3] = fotoelectrones_vuv_map[1]

  image[:, :, :, 4] = tiempos_vis_map[0]
  image[:, :, :, 5] = tiempos_vis_map[1]
  image[:, :, :, 6] = tiempos_vuv_map[0]
  image[:, :, :, 7] = tiempos_vuv_map[1]

  return image