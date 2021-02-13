import wave
from shutil import copyfile

import d3dshot
import numpy as np
import cv2  # OpenCV
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.cluster import KMeans
import os
import pyautogui
import time
from difflib import SequenceMatcher

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

# CHAR_DIM = 28
CHAR_DIM = 32
CHAR_DIM_W = 32
CHAR_DIM_H = 48
EPOCHS = 300

ALPHABET = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
            'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z',
            'x', 'c', 'v', 'b', 'n', 'm', '(', ')', '&', '!',
            'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
            'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z',
            'X', 'C', 'V', 'B', 'N', 'M',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '\'', 'Ø', '+', '-', ','
            ]

DIMS = [[690, 780, 490, 950], [690, 780, 1000, 1480], [800, 890, 490, 950], [800, 890, 1000, 1480]]


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image, kernel=np.ones((6, 6))):
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, kernel=np.ones((6, 6))):
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    # return cv2.resize(region, (CHAR_DIM_W, CHAR_DIM_H), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(region, (CHAR_DIM, CHAR_DIM), interpolation=cv2.INTER_CUBIC)


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    # ann.add(Dense(128, input_dim=CHAR_DIM_W * CHAR_DIM_H, activation='sigmoid'))
    ann.add(Dense(128, input_dim=CHAR_DIM ** 2, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=["categorical_accuracy", "accuracy", "mean_squared_error"])
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=False, )
    print("\nTraining completed...")
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for i in range(0, len(contours)):
        center1, size1, angle1 = cv2.minAreaRect(contours[i])
        width1, height1 = size1
        if hierarchy[0][i][3] == 0:
            x, y, w, h = cv2.boundingRect(contours[i])
            region = image_bin[y:y + h + 1, x:x + w + 1]
            # cv2.imshow("woaow", region)
            # cv2.waitKey()

            region = dilate(region)

            # kernel = np.ones((3, 3))
            # kernel = np.asarray([
            #     [0, 1, 1, 0],
            #     [1, 1, 1, 1],
            #     [1, 1, 1, 1],
            #     [0, 1, 1, 0],
            # ],
            #     np.uint8
            # )
            # region = erode(region, kernel)

            # kernel = np.asarray([
            #     [1, 1, 1, 1, 1, 1],
            #     [1, 1, 1, 1, 1, 1],
            #     [0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0],
            #     [1, 1, 1, 1, 1, 1],
            #     [1, 1, 1, 1, 1, 1]
            # ],
            #     np.uint8
            # )
            # kernel = np.asarray([
            #     [0, 0, 0, 0],
            #     [0, 1, 1, 0],
            #     [0, 1, 1, 0],
            #     [0, 0, 0, 0],
            # ],
            #     np.uint8
            # )
            # region = erode(region, kernel)
            # region = dilate(region, kernel)

            # region = erode(region)

            # cv2.imshow("woaow", region)
            # cv2.waitKey()
            # region = erode(region)
            region = resize_region(region)
            regions_array.append([region, (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.imshow("woaow", region)
            # cv2.waitKey()

    first_row = []
    second_row = []

    for ra in regions_array:
        flag = False
        for r1 in regions_array:
            if ra[1][1] > r1[1][1] + r1[1][3]:
                second_row.append(ra)
                flag = True
                break
        if not flag:
            first_row.append(ra)

    first_row = sorted(first_row, key=lambda x: x[1][0])
    second_row = sorted(second_row, key=lambda x: x[1][0])

    regions_array = first_row + second_row

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # x_next - (x_current + w_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances, contours


def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    # print(max(enumerate(k_means.cluster_centers_), key=lambda x: x[1]))
    # print(k_means.cluster_centers_)
    # print(k_means.labels_)
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def image_grayy(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_binn(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    # ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_most_similar(correct, answers):
    max = similar(correct, answers[0])
    maxText = answers[0]
    index = 0
    for i in range(1, 4):
        current = similar(correct, answers[i])
        if current > max:
            max = current
            maxText = answers[i]
            index = i
    return max, maxText, index


def prepare_img_for_roi(img):
    img_bin = image_binn(image_grayy(img))
    # cv2.imshow("W", img_bin)
    # cv2.waitKey()
    # kernel = np.asarray([
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     ],
    #     np.uint8
    # )
    # kernel = np.asarray([
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    # ],
    #     np.uint8
    # )
    # kernel = np.asarray([
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #     ],
    #     np.uint8
    # )
    kernel = np.asarray([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
        np.uint8
    )
    # img_bin = erode(img_bin, kernel=np.ones((7, 5)))
    img_bin = erode(img_bin, kernel)
    # img_bin = dilate(img_bin, kernel=np.ones((3, 3)))
    # cv2.imshow("W", img_bin)
    # cv2.waitKey()
    return img_bin


def train_and_savemodel():
    dirname = os.path.dirname(__file__)

    filenameproba = os.path.join(dirname, 'songpop_text/all.jpg')
    print(filenameproba)

    image_color = load_image(filenameproba)

    img_bin = prepare_img_for_roi(image_color)

    selected_regions, letters, region_distances, contours = select_roi_with_distances(image_color.copy(), img_bin)
    print(selected_regions)
    print(letters)
    print(region_distances)
    print(contours)

    print("Broj prepoznatih regiona: ", len(letters))
    display_image(selected_regions)
    imgg = selected_regions.copy()
    cv2.drawContours(imgg, contours, -1, (255, 0, 0), 1)

    print(len(ALPHABET))
    inputs = prepare_for_ann(letters)
    outputs = convert_output(ALPHABET)
    ann = create_ann(output_size=len(ALPHABET))
    ann = train_ann(ann, inputs, outputs, epochs=EPOCHS)

    ann.save("./TEXT_MODEL_28.h5")


class OCR:

    def __init__(self, model=None, percent=0.5):
        self.percent = percent
        self.model = keras.models.load_model(model)
        self.d3d = d3dshot.create()

    def extract_green(self, img):
        # cv2.imshow("imag_hsv", img)
        # cv2.waitKey()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # cv2.imshow("imag_hsv", img_hsv)
        # cv2.waitKey()

        hsv_color1 = np.asarray([40, 40, 40])
        hsv_color2 = np.asarray([70, 255, 255])

        img = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
        # cv2.imshow("mask", img)
        # cv2.waitKey()
        img = np.expand_dims(img, axis=2)
        img = np.float32(img)
        return img

    def click_closest_answer(self, answer):
        self.detect_with_model(answer)

    def check_answer(self):
        pyautogui.click(x=727, y=718)
        im = self.d3d.screenshot()

        im = np.asarray(im)

        answers = []

        sums = []

        for i in range(4):
            slice = im[DIMS[i][0]:DIMS[i][1], DIMS[i][2]:DIMS[i][3]].copy()
            # display_image(X[i])

            green = self.extract_green(slice)
            sums.append(np.sum(green))
            scale_percent = 182.95  # percent of original size
            width = int(slice.shape[1] * scale_percent / 100)
            height = int(slice.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(slice, dim, interpolation=cv2.INTER_CUBIC)
            # display_image(resized)

            eh = prepare_img_for_roi(resized)
            # cv2.imshow("www", eh)
            # cv2.waitKey()
            selected_regions1, letters1, distances1, contours1 = select_roi_with_distances(resized.copy(), eh)
            print("Broj prepoznatih regiona: ", len(letters1))
            if i == 2: display_image(selected_regions1)

            # neophodno je da u K-Means algoritam bude prosledjena matrica u kojoj vrste odredjuju elemente
            # distances1 = np.array(distances1).reshape(len(distances1), 1)
            #
            # print(distances1)
            #
            # k_means = KMeans(n_clusters=2)
            # k_means.fit(distances1)
            #
            inputs1 = prepare_for_ann(letters1)
            results1 = self.model.predict(np.array(inputs1, np.float32))
            # print(display_result_with_spaces(results1, alphabet, k_means))
            # answers.append(display_result_with_spaces(results1, alphabet, k_means))
            res = ""
            for j, letter in enumerate(results1):
                res += ALPHABET[winner(letter)]
                if j < len(letters1) - 1 and distances1[j] > 7:
                    res += " "
            print(res)
            answers.append(res)
        maximus = 0
        i_maximus = -1
        for i, sum in enumerate(sums):
            if sum > maximus:
                i_maximus = i
                maximus = sum
        answer = answers[i_maximus]
        files = os.listdir("./smart")

        if len(files) == 0:
            try:
                copyfile("sample/sample.wav", "smart/" + answer + ".wav")
                print("CREATED")
            except Exception:
                print("ERROR")
            return

        file_names = [os.path.splitext(f)[0] for f in files]

        score, resultText, index = get_most_similar(answer, file_names)

        print("SCORE ", score)

        if score < 0.8:
            samplerate, data = wavfile.read("sample/sample.wav")
            print("Score is low")

            MSE = []
            THRESH = 1000
            for f in files:
                samplerate, data1 = wavfile.read(f)

                mse = ((data - data1) ** 2).mean()

                name = os.path.splitext(f)[0]

                MSE.append((name, mse, f))
            sorted(MSE, key=lambda x: x[1])
            minimum = MSE[0]
            maximum = MSE[len(MSE) - 1]

            print(minimum, maximum)
            mse = minimum[1]
            if mse < THRESH:
                print("BELLOW THRESH")
                new_file = minimum[0] + "--" + answer + ".wav"
                print("new name", new_file)
                try:
                    copyfile("smart/" + minimum[2], "smart_processed/" + new_file)
                    print("SAVED")
                    os.remove("smart/" + minimum[2])
                    print("DELETED")
                    # TODO sacuvati u bazu
                except Exception:
                    print("ERROR")
            else:
                print("WOW NEW FILE")
                try:
                    copyfile("sample/sample.wav", "smart_processed/" + answer + ".wav")
                    print("SAVED NEW")
                except Exception:
                    print("ERROR")

    def detect_with_model(self, answer, ss=True, img_path=None):
        # time.sleep(1)
        t1 = time.time()
        answers = []
        if ss:

            imm = self.d3d.screenshot()

            im = np.asarray(imm)
        else:
            dirname = os.path.dirname(__file__)

            im = load_image(os.path.join(dirname, img_path))

        for i in range(4):

            curr = im[DIMS[i][0]:DIMS[i][1], DIMS[i][2]:DIMS[i][3]]
            # display_image(X[i])
            # cv2.imshow("www", curr)
            # cv2.waitKey()
            scale_percent = 182.95  # percent of original size
            width = int(curr.shape[1] * scale_percent / 100)
            height = int(curr.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(curr, dim, interpolation=cv2.INTER_CUBIC)
            # display_image(resized)
            # cv2.imshow("www", resized)
            # cv2.waitKey()

            eh = prepare_img_for_roi(resized)
            # cv2.imshow("www", eh)
            # cv2.waitKey()
            selected_regions1, letters1, distances1, contours1 = select_roi_with_distances(resized.copy(), eh)
            # print("Broj prepoznatih regiona: ", len(letters1))

            # neophodno je da u K-Means algoritam bude prosledjena matrica u kojoj vrste odredjuju elemente
            # distances1 = np.array(distances1).reshape(len(distances1), 1)
            #
            # print(distances1)
            #
            # k_means = KMeans(n_clusters=2)
            # k_means.fit(distances1)
            #

            inputs1 = prepare_for_ann(letters1)

            t1 = time.time()
            results1 = self.model.predict(np.array(inputs1, np.float32))
            print("ann  time", time.time() - t1)

            # print(display_result_with_spaces(results1, alphabet, k_means))
            # answers.append(display_result_with_spaces(results1, alphabet, k_means))
            res = ""
            for j, letter in enumerate(results1):
                res += ALPHABET[winner(letter)]
                if j < len(letters1) - 1 and distances1[j] > 7:
                    res += " "
            print("-> ", res)
            answers.append(res)

        solutions = answer.split("--")
        results = []
        for a in solutions:
            score, resultText, index = get_most_similar(a, answers)
            results.append((score, resultText, index))
        result = max(results, key=lambda x: x[0])
        index = result[2]
        resultText = result[1]
        score = result[0]
        print()
        print(score, resultText, " ==> ", answer)
        print()
        print("Score time", time.time() - t1)
        if score > self.percent:
            # cv2.imshow("Winner", X[index])
            # cv2.waitKey()
            pyautogui.moveTo(DIMS[index][2] + 10, DIMS[index][0] + 10)
            pyautogui.click()
            pyautogui.moveTo(100, 100)


def detect(answer, img_path=None):
    ocr = OCR("TEXT_MODEL_28.h5")

    ocr.detect_with_model(answer, img_path=img_path, ss=False)


if __name__ == '__main__':
    # input = naziv snimka u bazi koji je algoritam prepoznao
    input = "Rudimental"

    # train_and_savemodel()
    detect(input, img_path='songpop_text/rudimental.png')
