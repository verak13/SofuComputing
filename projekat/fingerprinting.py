import threading
import wave
from collections import Counter
from time import time

import numpy as np
import pyaudio
import sqlite3
import scipy.io
from scipy.io import wavfile
import random
import os
import operator

import hashlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
from operator import itemgetter

# sample rate
SAMPLE_RATE = 44100

# FFT window size
WINDOW_SIZE = 4096

# Minimum amplitude in spectrogram that is considered a peak
MIN_AMPLITUDE = 5

# Ratio by which window overlaps
OVERLAP_RATIO = 0.5

# Degree to which a fingerprint can be paired with its neighbors
FAN_VALUE = 15

# Number of cells around an amplitude peak in the spectrogram in order to be considered a spectral peak
PEAK_NEIGHBORHOOD_SIZE = 20

# Thresholds on how close or far fingerprints can be in time in order to be paired as a fingerprint
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

# Number of bits to throw away from the front of the SHA1 hash in the fingerprint calculation
FINGERPRINT_REDUCTION = 20


def fingerprint(channel_samples, sample_rate=SAMPLE_RATE,
                wsize=WINDOW_SIZE,
                wratio=OVERLAP_RATIO,
                fan_value=FAN_VALUE,
                amp_min=MIN_AMPLITUDE):

    # FFT the signal and extract frequency components
    array2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=sample_rate,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # apply log transform since specgram() returns linear array
    array2D = 10 * np.log10(array2D)
    array2D[array2D == -np.inf] = 0

    # find local maxima
    local_maxima = calculate_peaks(array2D, amp_min=amp_min)

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)


def calculate_peaks(array2D, amp_min=MIN_AMPLITUDE):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(array2D, footprint=neighborhood) == array2D
    background = (array2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = array2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp
    # print(peaks_filtered)

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    # print(frequency_idx, time_idx)
    return zip(frequency_idx, time_idx)

def generate_hashes(peaks, fan_value=FAN_VALUE):
    IDX_FREQ = 0
    IDX_TIME = 1

    peaks = sorted(peaks, key=itemgetter(1))

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                # take current & next peak frequency value
                freq1 = peaks[i][IDX_FREQ]
                freq2 = peaks[i + j][IDX_FREQ]

                # take current & next -peak time offset
                t1 = peaks[i][IDX_TIME]
                t2 = peaks[i + j][IDX_TIME]

                # get diff of time offsets
                t_delta = t2 - t1

                # check if delta is between min & max
                if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1("%s|%s|%s".encode('utf-8') % (
                        str(freq1).encode('utf-8'), str(freq2).encode('utf-8'), str(t_delta).encode('utf-8')))
                    yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)


def take_fingerprints(file, filename):
    print("start")

    wf = wave.open(file, 'rb')

    print(wf)
    print(wf.getframerate())
    print(wf.getnframes())
    print(wf.getparams())
    # print(wf.readframes(184320))
    sound_arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.uint8)
    print(sound_arr)

    samplerate, data = wavfile.read(file)
    print(samplerate, data)
    print(f"number of channels = {data.shape[1]}")
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    print("left channel: ", data[:, 0])
    print("right channel: ", data[:, 1])

    hashes = set()
    for i in range(wf.getnchannels()):
        channel_hashes = fingerprint(data[:, i], sample_rate=wf.getframerate())
        channel_hashes = set(channel_hashes)
        hashes |= channel_hashes
        # print(hashes)
    values = []
    for hash, offset in hashes:
        values.append((filename.split('.')[0], hash, offset))
    print(values)
    print(len(values))

    conn = sqlite3.connect('songs.db')
    c = conn.cursor()
    for value in values:
        print(value[0], value[1], value[2])
        query = "INSERT INTO fingerprints (song, hash, offset) VALUES (?, ?, ?)"
        c.execute(query, value)

    conn.commit()
    conn.close()


def do_hash():
    file = "sample/sample.wav"
    wf = wave.open(file, 'rb')

    samplerate, data = wavfile.read(file)
    hashes = set()
    for i in range(wf.getnchannels()):
        channel_hashes = fingerprint(data[:, i], sample_rate=wf.getframerate())
        channel_hashes = set(channel_hashes)
        hashes |= channel_hashes
        # print(hashes)
    conn1 = sqlite3.connect('songs.db')
    c1 = conn1.cursor()
    t1 = time()
    answer = ""

    hashes = list(hashes)
    print(len(hashes))
    for index in range(0, len(hashes), 1000):
        if index + 1000 > len(hashes):
            j = len(hashes)
        else:
            j = index + 1000
        h = [hashes[k][0] for k in range(index, j)]

        qmark = ", ".join(["?"] * len(h))

        c1.execute(
            'SELECT song FROM fingerprints WHERE hash in ({}) group by song order by count(hash) desc'.format(qmark), h)
        result = c1.fetchall()
        if len(result) > 0:
            print(result)
            answer = result[0][0]
            break

    # for hasha, offset in hashes:
    #     # c1.execute('Select song From fingerprints WHERE hash=? ORDER BY Difference(hash, ?) DESC', (hasha, hasha))
    #     c1.execute('SELECT song FROM fingerprints WHERE hash = ?', (hasha,))
    #     result = c1.fetchall()
    #     if len(result) > 0:
    #         print(result)
    #         answer = result[0][0]
    #         break
    conn1.close()

    # answer = max(answer_dict, key=lambda key: answer_dict[key])
    print("DB time", time() - t1)
    print(answer)
    return answer


if __name__ == '__main__':
    # take_fingerprints("start.wav", "numa numa ej")

    conn = sqlite3.connect('songs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE fingerprints (song, hash, offset)''')
    conn.close()
    i = 0
    for filename in os.listdir('data'):
        if filename.endswith(".wav"):
            print("================================================")
            print(i, os.path.join('data', filename))
            print("================================================")
            take_fingerprints(os.path.join('data', filename), filename)
            i += 1

    conn1 = sqlite3.connect('songs.db')
    c1 = conn1.cursor()
    # t = ('0603e28991b838e87a03')
    c1.execute('SELECT * FROM fingerprints WHERE song = ?', ('7 rings--Ariana Grande',))
    print(c1.fetchall())
    conn1.close()

    # print(int.from_bytes(b'\x02\x00\x00\x00\x00\x00\x00\x00', 'little'))
