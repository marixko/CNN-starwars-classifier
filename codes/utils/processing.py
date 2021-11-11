import os
import copy
import random
from tensorflow.python.types.core import Value
import torch 
import torchvision
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image


def color2gray(im, mode="numpy"):
    """[summary]

    Args:
        im ([type]): [description]
        mode (str, optional): [description]. Defaults to "numpy".

    Returns:
        [type]: [description]
    """

    if mode == "numpy": # RGB to grayscale
        im = np.array(im)
        aux = im[:,:,0]*0.2989 + im[:,:,1]*0.587 + im[:,:,2]*0.1140
    else: #  mode == "cv2"
        im=np.array(im)
        aux = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale

    im[:,:,0] = aux
    im[:,:,1] = aux
    im[:,:,2] = aux
    return im


def flip_image(im, direction, mode="numpy"):
    """[summary]

    Args:
        im ([type]): [description]
        direction ([type]): [description]
        mode (str, optional): [description]. Defaults to "numpy".

    Returns:
        [type]: [description]
    """

    aux = copy.deepcopy(im)
    if direction == "horizontal":
        if mode == "numpy":
            for m in range(3):
                for j in range(im.shape[0]):
                    aux[j,:,m] = im[-j-1,:,m]
                zip
        elif mode == "tensorflow":
            aux = tf.image.flip_up_down(im)

        elif mode == "torch":
            tensor = torch.from_numpy(im)
            aux = torch.flip(tensor, [0])

        else: # mode == cv2
            aux = cv2.flip(im, 0)

    if direction == "vertical":
        if mode == "numpy":
            for m in range(3):
                for j in range(im.shape[1]):
                    aux[:,j,m] = im[:,-j-1,m]

        elif mode == "tensorflow":
            aux = tf.image.flip_left_right(im)
        
        elif mode == "torch":
            tensor = torch.from_numpy(im)
            aux = torch.flip(tensor, [1])

        else: # mode == cv2
            aux = cv2.flip(im, 1)
    return aux



def random_contrast(im, mode="numpy"):
    factor = random.uniform(1,10)
    if mode == "numpy":
        aux = copy.deepcopy(im).astype(np.int16)
        for m in range(3):
            aux[:,:,m] = np.clip(128 + factor * (aux[:,:,m] - 128), 0 , 255).astype(np.uint8)
    
    elif mode == "tensorflow":
        aux = tf.image.random_contrast(im, 1, 10)

    elif mode == "torch":
        aux = torchvision.transforms.ColorJitter(contrast = (1,10))(im) 
    else: # mode == cv2
        aux = cv2.convertScaleAbs(im, alpha=factor, beta=0)
    return aux

#####################################################################
##  For the next function, we only use implementation from Keras   ##
#####################################################################

def random_rotation(im):
    layer = tf.keras.layers.RandomRotation(factor = (0,1))
    im = layer(im)
    return im

def random_zoom(im):
    vmin = random.uniform(0,1)
    vmax = random.uniform(vmin,1)
    layer = tf.keras.layers.RandomZoom(height_factor = (-vmin,-vmax))
    im = layer(im)
    return im

def random_brightness(im):
    im = tf.image.random_brightness(im,0.5)
    return im


def data_augmentation(im, n):
    im_aug = []
    for i in range(n):
        im_aug.append(random_rotation(im))
        im_aug.append(random_zoom(im))
        im_aug.append(random_brightness(im))
        im_aug.append(random_contrast(im, mode="tensorflow"))
    
    im_aug.append(flip_image(im, direction="vertical", mode="tensorflow"))
    im_aug.append(flip_image(im, direction="horizontal", mode="tensorflow"))
    
    return im_aug

def apply_filter(im, type="gaussian", ksize = np.array([6,6]), verbose=True):

    if type == "gaussian":
        try:
            im = cv2.GaussianBlur(im, ksize,0)
        except:
            if verbose:
                print("Odd number is required for the kernel size. Changing kernel size to:  "+ str(ksize+1))
            im = cv2.GaussianBlur(im, ksize+1,0)
    
    elif type == "blur":
        im = cv2.blur(im, ksize) 

    elif type == "median":
        try:
            im = cv2.medianBlur(im,ksize[0])
        except:
            if verbose:
                print("Odd number is required for the kernel size. Changing kernel size to:  "+ str(ksize[0]+1))
            im = cv2.medianBlur(im,ksize[0]+1)
    else:
        raise(ValueError)

    return im
    
