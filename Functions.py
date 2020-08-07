import numpy as np
from keras.preprocessing import image 
import cv2


def get_image_value(path, dim, img_type = 'normal'): 
    img = image.load_img(path, target_size =dim)
    img = image.img_to_array(img)
    if img_type =='grey':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(img.shape[0], img.shape[1], 1)
    
    return img/255

# def get_image_value(path, dim, img_type = 'normal'): 
#     if img_type =='grey':
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
#     else: 
#         img = cv2.imread(path)
#     return img/255

# def get_image_value(path, dim, img_type):
#     img = cv2.imread(path, cv2.COLOR_BGR2GRAY)

#     # convert to 3 equal channels
# #     img = cv2.merge((img, img, img))

# #     # create 1 pixel red image
# #     red = np.zeros((1, 1, 3), np.uint8)
# #     red[:] = (0,0,255)

# #     # create 1 pixel blue image
# #     blue = np.zeros((1, 1, 3), np.uint8)
# #     blue[:] = (255,0,0)

# #     # append the two images
# #     lut = np.concatenate((red, blue), axis=0)

# #     # resize lut to 256 values
# #     lut = cv2.resize(lut, (1,256), interpolation=cv2.INTER_CUBIC)

# #     # apply lut
# #     result = cv2.LUT(img, lut)
    
    
#     return img





def non_max_suppression(boxes, probs, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return the indexes of only the bounding boxes to keep
    return pick