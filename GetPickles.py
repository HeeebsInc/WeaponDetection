import pickle 
import pandas as pd 
import os
import numpy as np 
import cv2
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import Functions as func
import var



def get_img_array(img_paths, dim, img_type): 
    from tqdm import tqdm
    final_array = []
    for path in tqdm(img_paths): 
        img = func.get_image_value(path, dim, img_type)
        final_array.append(img)
    if img_type == 'grey':
        final_array = np.array(final_array)
        return final_array.reshape(final_array.shape[0], var.dimension[0],var.dimension[0],1)
    else: 
        return np.array(final_array)

def get_pickles(nn_type):
    from tqdm import tqdm
    df = pd.DataFrame(columns = ['path', 'label'])
    negative = [f'../FinalImages/Negative/{i}' for i in os.listdir('../FinalImages/Negative')]
    neg_labels = [0 for i in range(len(negative))]
    positive = [f'../FinalImages/Positive/{i}' for i in os.listdir('../FinalImages/Positive')]
    pos_labels = [1 for i in range(len(positive))]
    concat_path = negative[:len(positive)+ len(positive)] + positive
    concat_labels = neg_labels[:len(positive)+ len(positive)] + pos_labels


    df.path = concat_path 
    df.label = concat_labels 

    X = df
    y = df[['label']]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify = X.label, train_size = .95, random_state = 10)

    x_train.drop('label', axis = 1, inplace = True)
    x_test.drop('label', axis = 1, inplace = True)
    
    if nn_type == 'normal': 
        DIM =  var.normal_dimension 
    elif nn_type == 'mobilenet': 
        DIM = var.mobilenet_dimension
    
    elif nn_type == 'inceptionnet': 
        DIM = var.inception_dimension
        
    elif nn_type == 'vgg16': 
        DIM = var.vgg_dimension
        
    new_x_train = get_img_array(x_train.path.values, DIM, img_type = var.img_type)
    new_x_test = get_img_array(x_test.path.values, DIM, img_type = var.img_type)
    
    pickle.dump(new_x_train, open(f'../Pickles/{nn_type}_x_train.p', 'wb'), protocol=4)
    pickle.dump(y_train, open(f'../Pickles/{nn_type}_y_train.p', 'wb'), protocol=4)
    pickle.dump(new_x_test, open(f'../Pickles/{nn_type}_x_test.p', 'wb'), protocol=4)
    pickle.dump(y_test, open(f'../Pickles/{nn_type}_y_test.p', 'wb'), protocol=4)
    
    
    
def get_samples(nn_type): 
    x_train = pickle.load(open(f'../Pickles/{nn_type}_x_train.p', 'rb'))
    x_test = pickle.load(open(f'../Pickles/{nn_type}_x_test.p', 'rb'))
    y_train = pickle.load(open(f'../Pickles/{nn_type}_y_train.p', 'rb'))
    y_test = pickle.load(open(f'../Pickles/{nn_type}_y_test.p', 'rb'))
    return x_train, x_test, y_train, y_test