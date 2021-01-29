import math
import os
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
from PIL import Image
from PIL.Image import Image
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, MaxPool2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_core.python.keras.utils import Sequence

BATCH = 8
TRAIN = 86
VALIDATE = 87
EPOCHS = 200


def configure_cnn():
    model = Sequential()

    # # 380x140
    # model.add(Conv2D(16, (3, 3), (2, 2), input_shape=(140, 380, 3)))
    # model.add(LeakyReLU())
    # # ~190x70
    # model.add(Conv2D(16, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~90x30
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~40x15
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())
    #
    # # ~20x7
    # model.add(Conv2D(32, (3, 3), (2, 2)))
    # model.add(LeakyReLU())

    # 380x140
    model.add(Conv2D(16, (3, 3), (2, 2), input_shape=(140, 380, 1), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))
    # 380x140

    model.add(Conv2D(16, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))
    # 380x140

    model.add(Conv2D(16, (3, 3), (2, 2)))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    # 380x140
    model.add(Conv2D(32, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(32, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(32, (3, 3), (2, 2)))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    # 380x140
    model.add(Conv2D(64, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(64, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Conv2D(64, (3, 3), (2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(MaxPool2D((2, 2), strides=(1, 1), padding="same"))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dense(8))
    model.add(LeakyReLU())
    model.add(Dense(2))

    model.build()

    model.summary()

    return model


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def open_data(length, path="./data/"):
    x = np.empty((length, 140, 380, 1))
    y = np.empty((length, 2))
    classes = []
    with open(path + "classes.txt") as f:
        for line in f.readlines():
            try:
                index = int(line)
                classes.append(index)
            except Exception:
                pass

    for i in range(length):
        img = cv2.imread(path + "{}.png".format(i + 1))
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_color1 = np.asarray([20, 100, 100])  # white!
        hsv_color2 = np.asarray([30, 255, 255])  # yellow! note the order

        mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

        img = np.expand_dims(mask, axis=2)

        # cv2.imwrite("LMAO.png", mask)
        #
        # plt.imshow(mask)  # this colormap will display in black / white
        # plt.show()
        # sleep(1)
        # plt.imshow(rgb)
        # plt.show()
        # cv2.imshow("d", rgb)

        x[i] = img
        y[i] = np.zeros(2)
        y[i][classes[i]] = 1

    return x, y


def train_model():
    x, y = open_data(TRAIN, "./data/")
    x_val, y_val = open_data(VALIDATE, "./validate_data/")
    generator = Generator(x, y, BATCH)

    model = configure_cnn()

    model.compile(
        optimizer=Adam(learning_rate=0.03),
        loss=binary_crossentropy,
        metrics=[BinaryAccuracy(), MeanSquaredError()],
    )

    history = model.fit_generator(generator,
                                  validation_data=(x_val, y_val),
                                  steps_per_epoch=TRAIN // BATCH,
                                  shuffle=True,
                                  epochs=EPOCHS)

    model.save("MODEL")


def grab_interesting_area():
    # img = ImageGrab.grab(bbox=(760, 470, 1140, 610))
    img = ImageGrab.grab(bbox=(760, 520, 1140, 660))
    # img = ImageGrab.grab(bbox=(760, 570, 1140, 710))
    # img.show()
    return img


def generate_data(delay=0.3, duration=60, start_delay=3, path="./data/"):
    print("Starting recording in:")
    for i in range(start_delay):
        sleep(1)
        print(3 - i)

    sleep(1)
    index = 0
    time = 0
    while time < duration:
        # img = pyautogui.screenshot()

        # part of the screen
        img = grab_interesting_area()
        img.save(path + "{}.png".format(index))

        # img = np.asarray(img)
        sleep(0.3)
        time += delay
        index += 1


def sort(val):
    return int(val.split(".")[0])


def order_files_by_number(path):
    dir = os.listdir(path)
    dir2 = []
    for d in dir:
        try:
            int(d.split(".")[0])
            dir2.append(d)
        except Exception:
            pass
    dir2.sort(key=sort)
    print(dir2)

    for i, d in enumerate(dir2):
        img = Image.open(path + d)
        img.save(path + str(i) + ".png")


if __name__ == '__main__':
    # configure_cnn()
    # generate_data()
    # generate_data(path="./validate_data/")

    # dafuck("./validate_data/")
    train_model()
