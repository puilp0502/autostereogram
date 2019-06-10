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

def compute_diff(img, gap=None, nd=16):
    if gap == None:
       gap = offset(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
       print(gap)
    left = img[:, 0:img.shape[1]-gap, :]
    right = img[:, gap-1:-1, :]
    leftg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    rightg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=nd, blockSize=15)
    disparity = stereo.compute(leftg, rightg)
    return disparity

def autocorr(grey_img):
    corr = scipy.signal.fftconvolve(grey_img, np.fliplr(grey_img[:, 0:grey_img.shape[1]//6]), mode='valid')
    return corr

def convolve_normal(img, kernel_size=2):    
    new_img = np.zeros(img.shape)
    window = scipy.signal.windows.gaussian(kernel_size, kernel_size/9)
    kernel = np.outer(window, window)
    kernel /= np.sum(kernel)
    for i in range(img.shape[2]):
        new_img[:, :, i] = scipy.signal.fftconvolve(img[:, :, i], kernel, mode='same')
    return new_img



if __name__ == "__main__":
    cap = cv2.VideoCapture('datasets/blackisgood.webm')
    i = 0
    for i in range(350):
        ret, frame = cap.read()
    
    while True:
        print('decoding frame %d' % i)
        ret, frame = cap.read()
    
        try:
            cv2.imshow('orig', cv2.resize(frame, (1280, 720)))
            sol = compute_diff(frame.astype('uint8'), 255)
            print('sol shape', sol.shape)
            cv2.imshow('sol', sol/256)
            #plt.imshow(sol, cmap='gray') 
            #plt.show()
        except Exception as e:
            logging.exception('shit happended while decoding frame %d; skipping' % i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
