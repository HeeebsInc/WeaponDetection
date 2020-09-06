#normal 
import pickle 
import pandas as pd 
import os
import numpy as np 
import cv2
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PyFunctions import var
import random



def get_edged(img, dim): 
    '''This function will convert an image into an edged version using Gaussian filtering''' 
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    wide = cv2.Canny(blurred, 10,200)
    wide = cv2.resize(wide, dim, interpolation = cv2.INTER_CUBIC)
    return wide


def get_image_value(path, dim, edge = False): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used.  If edge is specified as true, it will pass the img array to get_edged which returns a filtered version of the img'''
    if edge == True: 
        img = cv2.imread(path)
        edged = get_edged(img, dim)
        return edged
    else: 
        img = image.load_img(path, target_size = dim)
        img = image.img_to_array(img)
        return img/255

def get_img_array(img_paths, dim, edge): 
    '''This fucntion takes a list of image paths and returns the np array corresponding to each image.  It also takes the dim and whether edge is specified in order to pass it to another function to apply these parameters.  This function uses get_image_value to perform these operations'''
    final_array = []
#     from tqdm import tqdm
#     for path in tqdm(img_paths):
    for path in img_paths:
        img = get_image_value(path, dim, edge)
        final_array.append(img)
    final_array = np.array(final_array)
    if edge:
        return final_array.reshape(final_array.shape[0], final_array.shape[1], final_array.shape[2], 1)
    else: 
        return final_array
        
def get_tts(nn_type, version = 1, edge = False, balance = False, pick = False):
    '''This function will creates a pickled file given the type of neural network architecture.  
    Using the Var.py file, the function will determine the specified dimension of the algorithm and create pickles given the NN type.  For this function to work, you must create a folder outside the repo called Pickles
    Version parameter corresponds to the type of train test split: 
        version = 1 --> using ROI and positives and hand dataset as negative
        version = 2 --> using positive and negative ROI
          
        edge --> corresponds to whether it should apply edge detection to the photos within the split'''
        
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

#Using Seperated ROI ang hand data 
    if version == 1:
        pistol_paths = [f'../Separated/FinalImages/Pistol/{i}' for i in os.listdir('../Separated/FinalImages/Pistol')] 
        pistol_labels = [1 for i in range(len(pistol_paths))]

        rifle_paths = [f'../Separated/FinalImages/Rifle/{i}' for i in os.listdir('../Separated/FinalImages/Rifle')] 
        rifle_labels = [2 for i in range(len(rifle_paths))]    

        neg_paths = [f'../hand_dataset/Neg/{i}' for i in os.listdir('../hand_dataset/Neg')]
        random.shuffle(neg_paths)
        neg_paths = neg_paths[:len(pistol_paths)- 500]
        neg_labels = [0 for i in range(len(neg_paths))]
        
    elif version == 2: 
        pistol_paths = [f'../Separated/FinalImages/Pistol/{i}' for i in os.listdir('../Separated/FinalImages/Pistol')] 
        pistol_labels = [1 for i in range(len(pistol_paths))]

        rifle_paths = [f'../Separated/FinalImages/Rifle/{i}' for i in os.listdir('../Separated/FinalImages/Rifle')] 
        rifle_labels = [2 for i in range(len(rifle_paths))]    

        neg_paths = [f'../Separated/FinalImages/NoWeapon/{i}' for i in os.listdir('../Separated/FinalImages/NoWeapon')]
        random.shuffle(neg_paths)
        neg_paths = neg_paths[:len(pistol_paths)- 500]
        neg_labels = [0 for i in range(len(neg_paths))]
        
        
    if balance == True: 
        random.shuffle(pistol_paths)
        pistol_paths = pistol_paths[:len(rifle_paths)+150]
        neg_paths = neg_paths[:len(rifle_paths)+150]
        
        pistol_labels = [1 for i in range(len(pistol_paths))]
        rifle_labels = [2 for i in range(len(rifle_paths))]
        neg_labels = [0 for i in range(len(neg_paths))]
    paths = pistol_paths + rifle_paths + neg_paths
    labels = pistol_labels + rifle_labels + neg_labels
    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90, random_state = 10)

    if edge == True:      
        new_x_train = get_img_array(x_train, DIM, edge = True)
        new_x_test = get_img_array(x_test, DIM, edge = True)
    else: 
        new_x_train = get_img_array(x_train, DIM, edge = False)
        new_x_test = get_img_array(x_test, DIM, edge = False)
    
    print('Train Value Counts')
    print(pd.Series(y_train).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Test Value Counts')
    print(pd.Series(y_test).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Train Shape')
    print(new_x_train.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Test Shape')
    print(new_x_test.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    tts = (new_x_train, new_x_test, y_train, y_test)
    if pick == True:
        if edge == True:
            pickle.dump(tts, open(f'../Pickles/edge_{nn_type}_tts.p', 'wb'), protocol=4)
        else:
            pickle.dump(tts, open(f'../Pickles/{nn_type}_tts.p', 'wb'), protocol=4)
    
    return tts

        
        
def get_samples(nn_type, edge = False): 
    '''After performing the get_pickles function above, this function can be used to retrieve the pickled files given a specific NN type.  '''
    if edge == True: 
        x_train, x_test, y_train, y_test = pickle.load(open(f'../Pickles/edge_{nn_type}_tts.p', 'rb'))
    
    else: 
        x_train, x_test, y_train, y_test = pickle.load(open(f'../Pickles/{nn_type}_tts.p', 'rb'))

    return x_train, x_test, y_train, y_test

def non_max_suppression(boxes, overlapThresh= .5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick
    return boxes[pick].astype("int")
    

def get_img_prediction_bounding_box(path, model, dim, edge = False):
    '''This function will create a bounding box over what it believes is a weapon given the image path, dimensions, and model used to detect the weapon.  Dimensions can be found within the Var.py file.  This function is still being used as I need to apply non-max suppresion to create only one bounding box'''
    img = get_image_value(path, dim, edge = edge)

    if edge == True:
        img = img.reshape(1, img.shape[0], img.shape[1], 1)
    else: 
        img = img.reshape(1, img.shape[0], img.shape[1], 3)
    
    pred = model.predict(img)[0]
    
    category_dict = {0: 'No Weapon', 1: 'Handgun', 2: 'Rifle'}
    cat_index = np.argmax(pred)
    cat = category_dict[cat_index]
    print(f'{path}\t\tPrediction: {cat}\t{int(pred.max()*100)}% Confident')
    
    
    #speed up cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(10)
    
    img = cv2.imread(path)

    clone = img.copy() 
    clone2 = img.copy()
    
#     if cat_index != 0:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
#     ss.switchToSelectiveSearchQuality()
    ss.switchToSelectiveSearchFast()

    rects = ss.process() 

    windows = []
    locations = []
    print(f'Creating Bounding Boxes for {path}')
    for x, y, w,h in rects[:1001]: 
        startx = x 
        starty = y 
        endx = x+w 
        endy = y+h 
        roi = img[starty:endy, startx:endx]
        if edge == True:
            roi = get_edged(roi, dim = dim)
        roi = cv2.resize(roi, dsize =dim, interpolation = cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))

    windows = np.array(windows)
    if edge == True:
        windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
    else: 
        windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 3)
    windows = np.array(windows)
    locations = np.array(locations)
    predictions = model.predict(windows)
    nms = non_max_suppression(locations)
    print(nms)
    bounding_cnt = 0
    for idx in nms:
        if np.argmax(predictions[idx]) != cat_index: 
            continue
#         print(np.argmax(predictions[idx]), predictions[idx].max())
        startx, starty, endx, endy = locations[idx]
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
        text = f'{category_dict[np.argmax(predictions[idx])]}: {int(predictions[idx].max()*100)}%'
        cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        bounding_cnt += 1

    if bounding_cnt == 0: 
        pred_idx= [idx for idx, i in enumerate(predictions) if np.argmax(i) == cat_index]
        cat_locations = np.array([locations[i] for i in pred_idx])
        nms = non_max_suppression(cat_locations)
        if len(nms)==0:
            cat_predictions = predictions[:,cat_index]
            pred_max_idx = np.argmax(cat_predictions)
            pred_max = cat_predictions[pred_max_idx]

            pred_max_window = locations[pred_max_idx]
            startx, starty, endx, endy = pred_max_window
            cv2.rectangle(clone, (startx, starty), (endx, endy),  (0,0,255),2)
            text = f'{category_dict[cat_index]}: {int(pred_max*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        for idx in nms: 
            startx, starty, endx, endy = cat_locations[idx]
            cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
            text = f'{category_dict[np.argmax(predictions[pred_idx[idx]])]}: {int(predictions[pred_idx[idx]].max()*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
               


    
#     pick = func.non_max_suppression(locations, probs = None)

#     for idx in pick: 
#         startx, startx, endx, endy = locations[idx]
#         cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
        
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    cv2.imshow(f'Test', np.hstack([clone, clone2]))
    cv2.waitKey(0)
    ss.clear()


    return predictions

def get_vid_frames(path, model, dim, edge = False): 
    '''This function will take a path to an mp4 file and return a list containing each frame of the video.  This function is used for creating bounding boxes within a video'''
    from tqdm import tqdm
    vid = cv2.VideoCapture(path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total = total_frames, desc = f'Splitting Video Into {total_frames} Frames')
    images = [] 
    sucess =1 
    while True: 
        try:
            success, img = vid.read() 
            img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
            images.append(img)
            pbar.update(1)
        except: 
            break
        

    pbar.close()
    images = np.array(images)
    
    return images



def window_prob_func(img, model, dim, edge):
    '''Given an image, this function will extract the segmented bounding boxes and return the ones with the rest ROI (using non_max_suppresion.  If there is no overlap, it will return None)'''
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
        if edge == True:
            roi = get_edged(roi, dim = dim)
        roi = cv2.resize(roi, dsize =dim, interpolation = cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))

    windows = np.array(windows)
    if edge == True:
        windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
    else: 
        windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 3)
    predictions = model.predict(windows)
    locations = np.array(locations)

    pick = non_max_suppression(locations, probs = None)
    for idx in pick: 
        prob = predictions[idx]
        if np.argmax(prob) == 0: 
            continue
        startx, startx, endx, endy = locations[idx]
        return (prob, (startx, starty, endx, endy))

        
    return None






