import threading
import wave

import pyaudio


import sqlite3

class RecordingObject:

    def __init__(self, chunk=1024, channels=2, fs=44100, record_handler=None):
        self.path = "test"
        self.filename = None
        self.data = []
        self.chunk_size = chunk
        self.channels = channels
        self.fs = fs
        self.sample_format = pyaudio.paInt16  # 16 bits per sample

        self.middle_man = MiddleMan(True)
        self.stream = None
        self.pyaudio = None

        self.record_handler = record_handler

    def record_sample(self):
        self.middle_man.set_condition(True)

        self.pyaudio = pyaudio.PyAudio()  # Create an interface to PortAudio

        index = 0
        for i in range(self.pyaudio.get_device_count()):
            print(type(i), i)
            try:
                info = self.pyaudio.get_device_info_by_index(i)
            except:
                continue
            print(info)
            if 'Stereo Mix' in info['name']:
                index = i
                print("Recording device index is {}".format(index))
                self.channels = info["maxInputChannels"]
                print(type(self.channels), self.channels)
                break
        print('Recording')

        # self.stream = self.pyaudio.open(format=self.sample_format,
        #                                 channels=self.channels,
        #                                 rate=self.fs,
        #                                 frames_per_buffer=self.chunk,
        #                                 input=True,
        #                                 input_device_index=index)
        self.stream = self.pyaudio.open(
                                        format=self.sample_format,
                                        channels=self.channels,
                                        rate=self.fs,
                                        # frames_per_buffer=self.chunk,
                                        input=True,
                                        # input_device_index=index
            )
        self.data = []  # Initialize array to store frames

        x = threading.Thread(target=self.__recording_thread)
        x.start()

        # print('Finished recording')
        #
        # # Save the recorded data as a WAV file


    def stop_recording(self, file):
        self.path = file
        self.middle_man.set_condition(False)

    def __recording_thread(self):
        while self.middle_man.check_condition():
            data = self.stream.read(self.chunk_size)
            self.data.append(data)
            # print("Recorded chunk")
        print("Stopped recording")
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio.terminate()
        print("Shut down pyaudio")

        self.record_handler.handle(self.data)

        wf = wave.open("sample/" + self.path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.data))
        wf.close()
        print("WRITTEN")


class MiddleMan:
    def __init__(self, value):
        self.message = value
        self.lock = threading.Lock()

    def check_condition(self):
        self.lock.acquire()
        value = self.message
        self.lock.release()
        return value

    def set_condition(self, value):
        self.lock.acquire()
        self.message = value
        self.lock.release()


if __name__ == '__main__':
    inp = ""

    rec = RecordingObject()
    rec.record_sample()
    while inp != "x":
        print("Enter 'x' to stop recording: ")
        inp = input()
    rec.stop_recording("test.wav")
