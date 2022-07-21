#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma,rescale_intensity
# Creates an image of original brain with segmentation overlay
def show_mask_on_image(im, mask, imsize=(10,20), alpha=0.7, cmaps=['gray', 'jet', 'spring', 'hot', 'brg']):

    plt.figure(figsize=imsize) # creat20e a figure with the default size 
    
    plt.subplot(1,2,1)
    plt.imshow(np.rollaxis(im,0), cmaps[0], interpolation='none')
    plt.subplot(1,2,2)

    plt.imshow(np.rollaxis(im,0),  cmaps[0], interpolation='none')
    for i,m in enumerate(mask):
        
        masked = np.ma.masked_where(m == 0, m)
        plt.imshow(np.rollaxis(masked,0), cmaps[i+1], interpolation='none', alpha=alpha)
    plt.show()
    
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
        
def show_list_images(imgs, title = None, titles=None,lsize=2,cmap='gray'):
    n_img = len(imgs)
    fig, m_axs = plt.subplots(1, n_img, figsize = (n_img*lsize, 2*lsize))
    if (title is not None):
        fig.suptitle(title, fontsize=16)
    i = 0
    for (c_im) in m_axs.T:
        c_im.imshow(imgs[i], cmap=cmap)
        c_im.axis('off')
        if (titles is not None):
            c_im.set_title(titles[i])
        i+=1

def show_n_images(imgs, titles = None, enlarge = 20, isaxis = True,
                  save_file = False, cmap='jet'):
    
    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)   
    
    fig1 = plt.figure() # create a figure with the default size 
    fig1.set_size_inches(enlarge, 2*enlarge)
    
    fname = 'image_'
    
    if not isaxis:
        plt.axis("off")
        
    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i]) 

        ax1.imshow(imgs[i], interpolation='none', origin='upper')
        if (titles is not None):
            ax1.set_title(titles[i])
            fname = fname + titles[i]
        #ax1.set_ylim(ax1.get_ylim()[::-1])

    if (save_file):
        file_name =  os.path.join(OUTPUT_DIR, fname + '.png')
        print ('Saving ', file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()
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