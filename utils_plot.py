import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh= lambda x: np.tanh(x)
binary_threshold = lambda x: 1 if x >= 0 else 0.
relu = lambda x: x if x >= 0 else 0.

#  Drow general function  
#  variation of: https://gist.github.com/zhmz1326/5a6cad0eae9479205506
def drow_function(func=sigmoid, func_name='Logistic Sigmoid Function',
                  fund_sampling = (-10, 10, 100), is_grid = False, is_ticks = True,
                 func_formula = r'$\sigma(x)=\frac{1}{1+e^{-x}}$'):


    y=np.linspace(*fund_sampling)
    plt.figure(figsize=(6,2), dpi=80, facecolor='w', edgecolor='k')
    fy=[func(z) for z in y]
    plt.plot(y,fy, 'b')
    if(is_grid):
        plt.grid()
    plt.xlabel('X ')
    plt.ylabel('Y')
    plt.title(func_name)

    if (not is_ticks):
        plt.xticks([], [])
        
    if(func_formula):
        plt.text(3,0.6,func_formula,fontsize=15)
    
    plt.show()


def drow_sigmoid():
    drow_function(func=sigmoid, func_name='Logistic Sigmoid Function')
    plt.show()
    
def drow_tanh():
    drow_function(func=tanh, func_name='Tanh Sigmoid Function',
                 func_formula = r'$tanh(x)=\frac{e^{x}-e^{x}}{e^{x}+e^{x}}$')
    plt.show()
    
def drow_binary_threshold():
    drow_function(func=binary_threshold, func_name='Binary threshold',
                 func_formula = None)
    plt.show()
#----------------------------------------------------------------------------
def show_n_images(imgs, cmap='gray', titles = None, enlarge = 10,
                  isaxis = True):
    
    plt.set_cmap(cmap)
    
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)   
    
    fig1 = plt.figure() # create a figure with the default size 
    fig1.set_size_inches(enlarge, 2*enlarge)
    
    if not isaxis:
        plt.axis("off")
        
         
    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i]) 

        ax1.imshow(imgs[i], interpolation='none')
        if (titles is not None):
            ax1.set_title(titles[i])
        #ax1.set_ylim(ax1.get_ylim()[::-1])

    plt.show()
#-----------------------------------------------------------------------------
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
def drow_roc_curve(ytest, ypred):
    
    fpr, tpr, _ = roc_curve(ytest, ypred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
#------------------------------------------------------------------------------
def drow_history(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+ metric])
    plt.title('model '+metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()    
#--------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def show_list_images_and_predictions(imgs, preds, title1 = 'image',title2 = 'prediction'):
    n_img = len(imgs)
    fig, m_axs = plt.subplots(2, n_img, figsize = (n_img*2, 4))
    i = 0
    for (c_im, c_lab) in m_axs.T:
        c_im.imshow(imgs[i])
        c_im.axis('off')
        c_im.set_title(title1)

        c_lab.imshow(y_train[ind])
        c_lab.axis('off')
        c_lab.set_title(title2) 
        i+=1
        
def show_list_images(imgs, titles=None):
    n_img = len(imgs)
    fig, m_axs = plt.subplots(1, n_img, figsize = (n_img*2, 4))
    i = 0
    for (c_im) in m_axs.T:
        c_im.imshow(imgs[i])
        c_im.axis('off')
        if (titles is not None):
            c_im.set_title(titles[i])
        i+=1

def plot_img_grid_array(imgs, n_col=4,n_row=4, titles=None):

    fig, axs = plt.subplots(n_row, n_col)
    l=0
    for i in range(n_col):
        for j in range(n_row):

            axs[i,j].imshow(imgs[:,:,l])
            if(titles is not None):
                axs[i,j].set_title(titles[l])
            axs[i,j].axis('off')
            l+=1

def plot_img_grid_list(imgs, n_col=4,n_row=4, titles=None):

    fig, axs = plt.subplots(n_row, n_col)
    l=0
    for i in range(n_col):
        for j in range(n_row):

            axs[i,j].imshow(imgs[l])
            if(titles is not None):
                axs[i,j].set_title(titles[l])
            axs[i,j].axis('off')
            l+=1