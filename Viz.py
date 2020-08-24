import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, confusion_matrix
import numpy as np
import pandas as pd
import itertools
import seaborn as sns




def plot_loss_accuracy(model_history): 
    train_loss = model_history.history['loss']
    train_acc = model_history.history['acc']
    test_loss = model_history.history['val_loss']
    test_acc = model_history.history['val_acc']
    epochs = [i for i in range(1, len(test_acc)+1)]

    fig, ax = plt.subplots(1,2, figsize = (15,5))
    ax[0].plot(epochs, train_loss, label = 'Train Loss')
    ax[0].plot(epochs, test_loss, label = 'Test Loss')
    ax[0].set_title('Train/Test Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label = 'Train Accuracy')
    ax[1].plot(epochs, test_acc, label = 'Test Accuracy')
    ax[1].set_title('Train/Test Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    

    
    
    
def plot_roc_auc(model, x_test, y_test): 
    fig, ax = plt.subplots(2,2, figsize = (18,10))
    #AUC CURVE
    y_test_prob = model.predict(x_test)

    y_test_precision, y_test_recall, spec = precision_recall_curve(y_test, y_test_prob)
    y_test_predict = np.where(y_test_prob >= .5, 1, 0).ravel()
    y_test_f1= f1_score(y_test, y_test_predict)
    y_test_auc = auc(y_test_recall, y_test_precision)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    ax[0,0].plot(y_test_recall, y_test_precision, marker='.', label='CNN')
    ax[0,0].plot([0, 1], [no_skill, no_skill], linestyle='--', label='50/50', color = 'Black')
    ax[0,0].set_xlabel('Recall')
    ax[0,0].set_ylabel('Precision')
    ax[0,0].set_title(f'AUC Curve')
    ax[0,0].legend()

    #ROC CURVE
    ns_probs = [0 for i in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    y_test_roc = roc_auc_score(y_test, y_test_prob)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    y_test_fpr, y_test_tpr, threshold = roc_curve(y_test, y_test_prob)
    ax[0,1].plot(ns_fpr, ns_tpr, linestyle='--', label='50/50')
    ax[0,1].plot(y_test_fpr, y_test_tpr, marker='.', label='CNN')
    ax[0,1].set_xlabel('False Positive Rate')
    ax[0,1].set_ylabel('True Positive Rate')
    ax[0,1].set_title(f'ROC Curve')
    ax[0,1].legend()
    
    
    
    df = pd.DataFrame({'Threshold': threshold, 'FPR': y_test_fpr, 'TPR': y_test_tpr})
    plt.plot(df.Threshold, df.FPR, label = 'False Positive Rate')
    plt.plot(df.Threshold, df.TPR, label = 'True Positive Rate')
    plt.xlabel('Threshold')
    plt.ylabel('FPR/TPR')
    plt.title('Change in FPR/TPR with Threshold')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame({'F1 Score': round(y_test_f1, 3), 'AUC': round(y_test_auc, 3), 'ROC':round(y_test_roc, 3)}, index = [0])


def plot_model_cm(y_test,y_train, y_train_prob, y_test_prob,thresholds, classes,
                          cmap=plt.cm.Blues):
    fig, ax = plt.subplots(len(thresholds),2, figsize = (10,10))

    for idx, thresh in enumerate(thresholds):
        y_test_predict = np.where(y_test_prob >= thresh, 1, 0)
        y_train_predict = np.where(y_train_prob >= thresh, 1, 0)
        train_cm = confusion_matrix(y_train, y_train_predict) 
        test_cm = confusion_matrix(y_test, y_test_predict)
        
        #test confusion
        ax[idx, 0].imshow(test_cm,  cmap=plt.cm.Blues) 

        ax[idx, 0].set_title(f'Test: Confusion Matrix | Threshold: {thresh}')
        ax[idx, 0].set_ylabel('True label')
        ax[idx, 0].set_xlabel('Predicted label')

        class_names = classes 
        tick_marks = np.arange(len(class_names))
        ax[idx, 0].set_xticks(tick_marks)
        ax[idx,0].set_xticklabels(class_names)
        ax[idx, 0].set_yticks(tick_marks)
        ax[idx, 0].set_yticklabels(class_names)

        th = test_cm.max() / 2. 

        for i, j in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
                ax[idx, 0].text(j, i, f'{test_cm[i, j]}',# | {int(round(test_cm[i,j]/test_cm.ravel().sum(),5)*100)}%',
                         horizontalalignment='center',
                         color='white' if test_cm[i, j] > th else 'black')
        ax[idx, 0].set_ylim([-.5,1.5])
        
        #TRAIN CONFUSION
        ax[idx, 1].imshow(train_cm,  cmap=plt.cm.Blues) 

        ax[idx, 1].set_title(f'Train: Confusion Matrix | Threshold: {thresh}')
        ax[idx, 1].set_ylabel('True label')
        ax[idx, 1].set_xlabel('Predicted label')

        class_names = classes 
        tick_marks = np.arange(len(class_names))
        ax[idx, 1].set_xticks(tick_marks)
        ax[idx,1].set_xticklabels(class_names)
        ax[idx, 1].set_yticks(tick_marks)
        ax[idx, 1].set_yticklabels(class_names)


        th = train_cm.max() / 2. 

        for i, j in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
                ax[idx, 1].text(j, i, f'{train_cm[i, j]}',# | {int(round(train_cm[i,j]/train_cm.ravel().sum(),5)*100)}%',
                         horizontalalignment='center',
                         color='white' if train_cm[i, j] > th else 'black')
        ax[idx, 1].set_ylim([-.5,1.5])
    plt.tight_layout()
    plt.show()


    


