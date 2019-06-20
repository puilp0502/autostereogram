import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from mevid import imread_rgb, rgb2gray, compute_diff, image_entropy, compute_entropy, offset3

if __name__ == "__main__":
    matplotlib.use('GTK3Cairo')

    img = imread_rgb('datasets/brahh.jpg')
    if img is None:
        logging.error("Error: file not found or unrecognized format")

    print('calculating entropy...')
    ef = plt.figure()
    g, e_g = compute_entropy(img)
    plt.plot(g, e_g)
#    gap = offset3(img)
#    print('estimated gap: %d' % gap)

    fig, ax = plt.subplots()
    g0 = 24
    dmap = compute_diff(img, g0)
    ai = plt.imshow(dmap, cmap='gray')

    axgap = plt.axes([0.25, 0.1, 0.65, 0.03])

    sgap = Slider(axgap, 'Gap', 0, 600, valinit=g0, valstep=1)

    def update(val):
        gap = int(sgap.val)
        dmap = compute_diff(img, gap, nd=32)
        entropy = image_entropy(dmap)
        ai.set_data(dmap)
        fig.suptitle('entropy = %f' % entropy)
        fig.canvas.draw_idle()

    sgap.on_changed(update)

    plt.show()
