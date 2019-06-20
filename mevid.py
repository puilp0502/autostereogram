import logging
from magiceye_solve import magiceye_solver
import cv2
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# matplotlib.use('GTK3Cairo')

def imread_grey(fname):
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)

def imread_rgb(fname):
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def offset2(img):
    corr = autocorr(img)[0]
    N = len(corr)
    T = 1
    fft = np.abs(np.fft.fft(corr)[1:N//2])
    freq = np.fft.fftfreq(N, T)[1:N//2]
    return 1 / freq[np.argmax(fft)], freq, fft

def image_entropy(img):
    """
    Calculates entropy of the greyscale image `img`.
    img: greyscale image, scaled to 0~255 range
    """
    # implements matlab entropy(I)-esque function
    p, bin_edges = np.histogram(img, 256, (0, 255))
    p = p[p>0]
    p = p / p.size
    logp = np.log2(p)
    return -1 * np.sum(p * logp)

def compute_entropy(img):
    entropy_history = []
    for i in range(32, img.shape[1]//4):
        print('computing entropy for gap %d' % i)
        dmap = compute_diff(img, i, nd=16)
#        dmap_range = np.max(dmap) - np.min(dmap)
#        dmap = (dmap - np.min(dmap)) * (255 / dmap_range)
        entropy_history.append(image_entropy(dmap))
    return np.arange(32, img.shape[1]//4), entropy_history

def offset3(img):
    g, f_g = compute_entropy(img)
    x = np.argmin(f_g)
    return g[0] + x

def offset(img):
    """
    calculates the offset that defines the stereoscopic effect
    """
    img = img - img.mean()
    ac = scipy.signal.fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    
    ac = ac[int(len(ac)/2)]
    idx = np.where((ac - np.median(ac)) / ac.std() > 3)[0]
    diffs = []
    diffs = np.ediff1d(idx)
    return np.max(diffs)

def compute_diff(img, gap=None, nd=16, md=0):
    if gap == None:
       gap = offset(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
       print(gap)
    if len(img.shape) == 3: # rgb
        left = img[:, 0:img.shape[1]-gap, :]
        right = img[:, gap-1:-1, :]
        leftg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    else: # greyscale
        leftg = img[:, 0:img.shape[1]-gap]
        rightg = img[:, gap-1:-1]
    print(leftg.shape)
    print(rightg.shape)
    stereo = cv2.StereoBM_create(numDisparities=nd, blockSize=19)
    stereo.setMinDisparity(md)
    disparity = stereo.compute(leftg, rightg)
    return disparity

def autocorr(grey_img, divisor=6):
    corr = scipy.signal.fftconvolve(grey_img, np.fliplr(grey_img[:, 0:grey_img.shape[1]//int(divisor)]), mode='valid')
    return corr

def convolve_normal(img, kernel_size=2):    
    new_img = np.zeros(img.shape)
    window = scipy.signal.windows.gaussian(kernel_size, kernel_size/9)
    kernel = np.outer(window, window)
    kernel /= np.sum(kernel)
    for i in range(img.shape[2]):
        new_img[:, :, i] = scipy.signal.fftconvolve(img[:, :, i], kernel, mode='same')
    return new_img
