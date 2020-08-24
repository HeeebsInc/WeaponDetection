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
from keras.utils import to_categorical



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
    pistol_paths = [f'../Separated/Pistol/{i}' for i in os.listdir('../Separated/Pistol')] + \
    [f'../Separated/Stock_Pistol/{i}' for i in os.listdir('../Separated/Stock_Pistol')]
    
    pistol_labels = [1 for i in range(len(pistol_paths))]
    
    rifle_paths = [f'../Separated/AR/{i}' for i in os.listdir('../Separated/AR')] + \
    [f'../Separated/Stock_AR/{i}' for i in os.listdir('../Separated/Stock_AR')]
    rifle_labels = [2 for i in range(len(rifle_paths))]
    

    negative = [f'../hand_dataset/Combined/{i}' for i in os.listdir('../hand_dataset/Combined')][:len(pistol_paths)]
    neg_labels = [0 for i in range(len(negative))]

    paths = pistol_paths + rifle_paths + negative
    labels = pistol_labels + rifle_labels + neg_labels


    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90)

    if nn_type == 'normal': 
        DIM =  var.norm_dimension 
    elif nn_type == 'mobilenet': 
        DIM = var.mobilenet_dimension
    
    elif nn_type == 'inceptionnet': 
        DIM = var.inception_dimension
        
    elif nn_type == 'vgg16': 
        DIM = var.vgg_dimension
        
    new_x_train = get_img_array(x_train, DIM, img_type = var.img_type)
    new_x_test = get_img_array(x_test, DIM, img_type = var.img_type)
    
    print(pd.Series(y_train + y_test).value_counts())
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    
    pickle.dump(new_x_train, open(f'../Pickles/{nn_type}_x_train.p', 'wb'), protocol=4)
    pickle.dump(y_train, open(f'../Pickles/{nn_type}_y_train.p', 'wb'), protocol=4)
    pickle.dump(new_x_test, open(f'../Pickles/{nn_type}_x_test.p', 'wb'), protocol=4)
    pickle.dump(y_test, open(f'../Pickles/{nn_type}_y_test.p', 'wb'), protocol=4)
    
    
    
def get_samples(nn_type): 
    x_train = pickle.load(open(f'../Pickles/{nn_type}_x_train.p', 'rb'))
    x_test = pickle.load(open(f'../Pickles/{nn_type}_x_test.p', 'rb'))
    y_train = pickle.load(open(f'../Pickles/{nn_type}_y_train.p', 'rb'))
    y_test = pickle.load(open(f'../Pickles/{nn_type}_y_test.p', 'rb'))
    
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    return x_train, x_test, y_train, y_test