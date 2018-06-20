import cv2
from matplotlib import pyplot as plt

from utils import *
from review import files


def test():

    image = read_image(files['jpg1'])
    debug(image, 'image')

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)
    plt.show()

    plt.imshow(image)
    plt.show()

    plt.imshow(cv2.cvtColor(excess_green(image), cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(excess_red(image), cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    test()