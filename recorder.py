#! /usr/bin/env python3
from utils import *


def record_old():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = NUM_NEW_CLIPS * MIN_CLIP_DURATION + 2.0

    LONG_STRING = "She had your dark suit in greasy wash water all year. Don't ask me to carry an oily rag like that!"

    print("Seak something \n Refrence sentence:", LONG_STRING)
    print("recording in 3 seconds")

    time.sleep(3) 
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


    print("* recording")
    print("PRESS CTRL-C to stop recording")
    frames = []

    #for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #try:
    #    while True:
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
    #except KeyboardInterrupt:
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(ENROLL_RECORDING_FNAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def split_recording(recording=ENROLL_RECORDING_FNAME):
    wav, sr = librosa.load(recording)
    total_duration = int(librosa.core.get_duration(wav))
    print(total_duration)
    print(ENROLL_RECORDING_FNAME)
    all_x = []
    all_sr = []
    for offset in range(0, total_duration, int(MIN_CLIP_DURATION)):
        x, sr = librosa.load(recording, sr=None, offset=offset,
                             duration= MIN_CLIP_DURATION)
        
        all_x.append(x)
        all_sr.append(sr)

    return get_stft(all_x[:-1])

if __name__ == "__main__":
    stfts = split_recording(ENROLL_RECORDING_FNAME)
    print(len(stfts))
    print(stfts[0].shape)
