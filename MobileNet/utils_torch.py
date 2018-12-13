import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(y_true, y_pred, classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    print('Classification Report')
    target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    print(classification_report(y_true, y_pred, target_names=target_names))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def save_cm_figs(y_true, y_pred, arc, target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']):
    # Compute confusion matrix

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(y_true, y_pred, classes=target_names,
                          title='Confusion matrix, without normalization')
    plt.savefig('./result/conf_no_norm_{}.png'.format(arc), bbox_inches='tight')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(y_true, y_pred, classes=target_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('./result/conf_norm_{}.png'.format(arc), bbox_inches='tight')
    plt.show()


def save_process_fig(train_pres, val_pres, arc):
    plt.figure()
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.plot(train_pres)
    plt.plot(val_pres)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./result/acc_{}.png'.format(arc), bbox_inches='tight')
    plt.show()
    # summarize history for loss
    """
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss cv:{0}'.format(idx + 1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./result/loss_cv{0}.png'.format(idx + 1), bbox_inches='tight')
    plt.show()
    """


def plot_cm(classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
            normalize=False,
            title='Confusion matrix',
            cmap=plt.cm.Blues):
    """
    cm = np.array([[20, 2, 3, 0, 3, 1, 0],
                   [6, 32, 4, 2, 4, 0, 0],
                   [16, 8, 70, 2, 17, 10, 0],
                   [0, 2, 0, 12, 0, 0, 0],
                   [8, 2, 18, 6, 68, 9, 1],
                   [12, 9, 57, 14, 97, 463, 9],
                   [1, 2, 0, 0, 0, 0, 12]])
    """
    cm = np.array([[ 20,   2,   3,   1,   3,   0,   0],
 [  5,  32,   1,   6,   4,   0,   0],
 [ 14,   7,  68,  10,  17,   7,   0],
 [  2,   1,   0,  10,   0,   0,   1],
 [ 11,   4,  18,   7,  64,   7,   1],
 [ 17,  10,  51,  11, 109, 455,   8],
 [  1,   2,   0,   0,   0,   0,  12]])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./result/conf_no_norm_{}.png'.format('ResNet_Weight'), bbox_inches='tight')


plot_cm()