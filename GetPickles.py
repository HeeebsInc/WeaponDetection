import Functions as func
import numpy as np 

def get_img_array(img_paths, dim, img_type, edge): 
    from tqdm import tqdm
    final_array = []
    for path in tqdm(img_paths): 
        img = func.get_image_value(path, dim, img_type, edge)
        final_array.append(img)
    final_array = np.array(final_array)
    if edge:
        return final_array.reshape(final_array.shape[0], dim[0], dim[1], 1)
    else: 
        return final_array