from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam
from keras import regularizers
import cv2
import numpy as np
from keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries

def get_conv_model(dim = (96,96), weights_path = 'NN_Weapon_Detection/FlaskApp/CNN-ModelCheckpointWeights3.h5'):
    inp_shape = dim
    act = 'relu'
    drop = .25
    kernal_reg = regularizers.l1(.001)
    dil_rate = 2
    optimizer = Adam(lr=.001)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation=act, input_shape=inp_shape,
                     kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same', name='Input_Layer'))
    #     model.add(Dense(64, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))

    model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same'))
    #     model.add(Dense(64, activation = 'relu'))
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
    model.load_weights(weights_path)
    return model

def get_img_prediction_bounding_box(path):
    dim = (96, 96, 3)
    model = get_conv_model(dim)
    img = get_image_value(path, dim)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    pred = model.predict(img)[0]

    category_dict = {0: 'No Weapon', 1: 'Handgun', 2: 'Rifle'}
    cat_index = np.argmax(pred)
    cat_prob = round(pred[cat_index], 2)
    cat = category_dict[cat_index]

    print(f'{path}\t\t{cat_index}||{cat}\t\t{pred.max()}\t\t{pred}')

    img = cv2.imread(path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    windows = []
    locations = []
    for x, y, w, h in rects:
        startx = x
        starty = y
        endx = x + w
        endy = y + h
        roi = img[starty:endy, startx:endx]
        roi = cv2.resize(roi, dsize=(96,96), interpolation=cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))

    windows = np.array(windows)

    predictions = model.predict(windows)

    clone = img.copy()
    clone2 = img.copy()
    cat_predictions = predictions[:, cat_index]
    pred_max_idx = np.argmax(cat_predictions)
    pred_max = cat_predictions[pred_max_idx]

    pred_max_window = locations[pred_max_idx]
    startx, starty, endx, endy = pred_max_window
    cv2.rectangle(clone, (startx, starty), (endx, endy), (0, 0, 255), 2)

    text = f'{cat}'
    cv2.putText(clone, text, (startx, starty), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 2)

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

