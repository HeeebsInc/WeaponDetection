from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam
from keras import regularizers
from keras.applications import MobileNetV2
import cv2
import numpy as np
from keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries


def get_conv_model(dim = (150,150, 3), model_weights = 'NN_Weapon_Detection/FlaskApp/V2_NoEdge_NoAugmentation.h5'):
    '''This function will create and compile a CNN given the input dimension'''
    inp_shape = dim
    act = 'relu'
    drop = .25
    kernal_reg = regularizers.l1(.001)
    optimizer = Adam(lr=.0001)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation=act, input_shape=inp_shape,
                     kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same', name='Input_Layer'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))

    model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))

    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(3, activation='softmax', name='Output_Layer'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.load_weights(model_weights)
    return model


def get_mobilenet(dim = (224,224, 3), model_weights = 'NN_Weapon_Detection/FlaskApp/Mobilenet.h5'):
    '''This function will create, compile and return the mobilenet neural network given the input dimensions.  '''
    model = Sequential()
    optimizer = Adam(lr=.0005)
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=dim))

    model.add(baseModel)
    headModel = model.add(AveragePooling2D(pool_size=(7, 7)))
    headModel = model.add(Flatten(name="flatten"))
    headModel = model.add(Dense(256, activation="relu"))
    headModel = model.add(Dropout(0.3))
    headModel = model.add(Dense(3, activation="softmax", name='Output'))

    for layer in baseModel.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.load_weights(model_weights)
    return model


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

def get_img_prediction_bounding_box(path, dim, model, edge = False):
    img = get_image_value(path, dim, edge=edge)

    if edge == True:
        img = img.reshape(1, img.shape[0], img.shape[1], 1)
    else:
        img = img.reshape(1, img.shape[0], img.shape[1], 3)

    pred = model.predict(img)[0]

    category_dict = {0: 'No Weapon', 1: 'Handgun', 2: 'Rifle'}
    cat_index = np.argmax(pred)
    cat = category_dict[cat_index]
    cat_prob = int(pred.max()*100)
    print(f'{path}\t\tPrediction: {cat}\t{int(pred.max() * 100)}% Confident')

    # speed up cv2
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
    for x, y, w, h in rects[:1001]:
        startx = x
        starty = y
        endx = x + w
        endy = y + h
        roi = img[starty:endy, startx:endx]
        if edge == True:
            roi = func.get_edged(roi, dim=dim)
        roi = cv2.resize(roi, dsize=dim, interpolation=cv2.INTER_CUBIC)
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
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)
        text = f'{category_dict[np.argmax(predictions[idx])]}: {int(predictions[idx].max() * 100)}%'
        cv2.putText(clone, text, (startx, starty + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        bounding_cnt += 1

    if bounding_cnt == 0:
        pred_idx = [idx for idx, i in enumerate(predictions) if np.argmax(i) == cat_index]
        cat_locations = np.array([locations[i] for i in pred_idx])
        nms = non_max_suppression(cat_locations)
        if len(nms) == 0:
            cat_predictions = predictions[:, cat_index]
            pred_max_idx = np.argmax(cat_predictions)
            pred_max = cat_predictions[pred_max_idx]

            pred_max_window = locations[pred_max_idx]
            startx, starty, endx, endy = pred_max_window
            cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)
            text = f'{category_dict[cat_index]}: {int(pred_max * 100)}%'
            cv2.putText(clone, text, (startx, starty + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        for idx in nms:
            startx, starty, endx, endy = cat_locations[idx]
            cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)
            text = f'{category_dict[np.argmax(predictions[pred_idx[idx]])]}: {int(predictions[pred_idx[idx]].max() * 100)}%'
            cv2.putText(clone, text, (startx, starty + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(img, model.predict, top_labels=5, hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,
                                                num_features=10, hide_rest=False)
    lime = mark_boundaries(temp/2 + .5, mask)
    return (cat_prob, cat, clone, lime)



def get_image_value(path, dim, edge = False, img_type = 'normal'):
    if edge == True:
        img = cv2.imread(path)
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

