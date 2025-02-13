import cv2
import numpy as np
from skimage import img_as_ubyte

def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def display_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()