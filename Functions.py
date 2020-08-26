#normal 
import pickle 
import pandas as pd 
import os
import numpy as np 
import cv2
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import var
from keras.utils import to_categorical
from lime import lime_image
from skimage.segmentation import mark_boundaries



def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    # return the edged image
    return edged


def get_image_value(path, dim, edge = False, img_type = 'normal'): 
    if edge == False: 
        img = cv2.imread('../TestImages/AR.jpg')
        blurred = cv2.GaussianBlur(img, (3,3), 0)
        wide = cv2.Canny(blurred, 10,200)
        tight = cv2.Canny(blurred, 225, 250)
        auto = auto_canny(blurred)
        wide = cv2.resize(wide, dim, interpolation = cv2.INTER_CUBIC)
        tight = cv2.resize(tight, dim, interpolation = cv2.INTER_CUBIC)
        auto = cv2.resize(auto, dim, interpolation = cv2.INTER_CUBIC)
        return tight
    else: 
        img = image.load_img(path, target_size = dim)
        img = image.img_to_array(img)
        if img_type =='grey':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            return img
        else: 
            return img/255

def get_pickles(nn_type, edge = False):
    
    if nn_type == 'normal': 
        DIM =  var.norm_dimension 
    elif nn_type == 'mobilenet': 
        DIM = var.mobilenet_dimension
    
    elif nn_type == 'inceptionnet': 
        DIM = var.inception_dimension
        
    elif nn_type == 'vgg16': 
        DIM = var.vgg_dimension
    elif nn_type == 'alexnet': 
        DIM = var.alex_dimension
        
    
    pistol_paths = [f'../Separated/FinalImages/Pistol/{i}' for i in os.listdir('../Separated/FinalImages/Pistol')] 


    pistol_labels = [1 for i in range(len(pistol_paths))]

    rifle_paths = [f'../Separated/FinalImages/Rifle/{i}' for i in os.listdir('../Separated/FinalImages/Rifle')] 
    rifle_labels = [2 for i in range(len(rifle_paths))]

    negative = [f'../Separated/FinalImages/NoWeapon/{i}' for i in os.listdir('../Separated/FinalImages/NoWeapon')][:len(pistol_paths)]
    neg_labels = [0 for i in range(len(negative))]


    paths = pistol_paths + rifle_paths + negative
    labels = pistol_labels + rifle_labels + neg_labels


    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90)

        
    new_x_train = get_img_array(x_train, DIM, img_type = var.img_type, edge = edge)
    new_x_test = get_img_array(x_test, DIM, img_type = var.img_type, edge = edge)
    
    print(pd.Series(y_train + y_test).value_counts())
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    if edge:
        pickle.dump(new_x_train, open(f'../Pickles/edge_{nn_type}_x_train.p', 'wb'), protocol=4)
        pickle.dump(y_train, open(f'../Pickles/edge_{nn_type}_y_train.p', 'wb'), protocol=4)
        pickle.dump(new_x_test, open(f'../Pickles/edge_{nn_type}_x_test.p', 'wb'), protocol=4)
        pickle.dump(y_test, open(f'../Pickles/edge_{nn_type}_y_test.p', 'wb'), protocol=4)
    else:
        pickle.dump(new_x_train, open(f'../Pickles/{nn_type}_x_train.p', 'wb'), protocol=4)
        pickle.dump(y_train, open(f'../Pickles/{nn_type}_y_train.p', 'wb'), protocol=4)
        pickle.dump(new_x_test, open(f'../Pickles/{nn_type}_x_test.p', 'wb'), protocol=4)
        pickle.dump(y_test, open(f'../Pickles/{nn_type}_y_test.p', 'wb'), protocol=4)
        
    
    
    
def get_samples(nn_type, edge = False): 
    if edge: 
        
        x_train = pickle.load(open(f'../Pickles/edge_{nn_type}_x_train.p', 'rb'))
        x_test = pickle.load(open(f'../Pickles/edge_{nn_type}_x_test.p', 'rb'))
        y_train = pickle.load(open(f'../Pickles/edge_{nn_type}_y_train.p', 'rb'))
        y_test = pickle.load(open(f'../Pickles/edge_{nn_type}_y_test.p', 'rb'))
    else: 
        x_train = pickle.load(open(f'../Pickles/{nn_type}_x_train.p', 'rb'))
        x_test = pickle.load(open(f'../Pickles/{nn_type}_x_test.p', 'rb'))
        y_train = pickle.load(open(f'../Pickles/{nn_type}_y_train.p', 'rb'))
        y_test = pickle.load(open(f'../Pickles/{nn_type}_y_test.p', 'rb'))
    
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    return x_train, x_test, y_train, y_test

def get_img_prediction_bounding_box(path, model, dim): 
    img = get_image_value(path, dim, var.img_type)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    pred = model.predict(img)[0]
    
    category_dict = {0: 'No Weapon', 1: 'Handgun', 2: 'Rifle'}
    cat_index = np.argmax(pred)
    cat = category_dict[cat_index]
    
    print(f'{path}\t\t{cat_index}||{cat}\t\t{pred.max()}\t\t{pred}')
    
    
    img = cv2.imread(path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process() 

    windows = []
    locations = []
    for x, y, w,h in rects: 
        startx = x 
        starty = y 
        endx = x+w 
        endy = y+h 
        roi = img[starty:endy, startx:endx]
        roi = cv2.resize(roi, dsize =dim, interpolation = cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))

    windows = np.array(windows)

    predictions = model.predict(windows)

    clone = img.copy()
    clone2 = img.copy()
    cat_predictions = predictions[:,cat_index]
    pred_max_idx = np.argmax(cat_predictions)
    pred_max = cat_predictions[pred_max_idx]
    
    pred_max_window = locations[pred_max_idx]
    startx, starty, endx, endy = pred_max_window
    cv2.rectangle(clone, (startx, starty), (endx, endy),  (0,0,255),2)

    text = f'{cat}'
    cv2.putText(clone, text, (startx, starty), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,255),2)
   
    
    cv2.imshow(f'{cat}', np.hstack([clone, clone2]))
    cv2.waitKey(0)






# def non_max_suppression(boxes, probs, overlapThresh=0.3):
#     # if there are no boxes, return an empty list
#     if len(boxes) == 0:
#         return []

#     # if the bounding boxes are integers, convert them to floats -- this
#     # is important since we'll be doing a bunch of divisions
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")

#     # initialize the list of picked indexes
#     pick = []

#     # grab the coordinates of the bounding boxes
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     # compute the area of the bounding boxes and grab the indexes to sort
#     # (in the case that no probabilities are provided, simply sort on the
#     # bottom-left y-coordinate)
#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = y2

#     # if probabilities are provided, sort on them instead
#     if probs is not None:
#         idxs = probs

#     # sort the indexes
#     idxs = np.argsort(idxs)
#     # keep looping while some indexes still remain in the indexes list
#     while len(idxs) > 0:
#         # grab the last index in the indexes list and add the index value
#         # to the list of picked indexes
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         # find the largest (x, y) coordinates for the start of the bounding
#         # box and the smallest (x, y) coordinates for the end of the bounding
#         # box
#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         # compute the width and height of the bounding box
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         # compute the ratio of overlap
#         overlap = (w * h) / area[idxs[:last]]

#         # delete all indexes from the index list that have overlap greater
#         # than the provided overlap threshold
#         idxs = np.delete(idxs, np.concatenate(([last],
#             np.where(overlap > overlapThresh)[0])))

#     # return the indexes of only the bounding boxes to keep
#     return pick



