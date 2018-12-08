#! /usr/bin/env python3
import librosa
import pyaudio
import time
import wave
from utils import *

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
ENROLL_RECORDING_FNAME = "output.wav"

LONG_STRING = "He determined to drop his litigation with the monastry, and relinguish his claim"+\
"s to the wood-cuting and fishery rihgts at once. He was the more"+\
 "ready to do this becuase the rights had becom much less valuable,"+\
 " and he had indeed the vaguest idea where the wood and river in quedtion were."

def record_old():
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
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()



def split_recording(recording=ENROLL_RECORDING_FNAME):
    wav, sr = librosa.load(recording)
    total_duration = int(librosa.core.get_duration(wav))
    all_x = []
    all_sr = []
    for offset in range(0, total_duration, int(MIN_CLIP_DURATION)):
        x, sr = librosa.load(recording, sr=None, offset=offset,
                             duration= MIN_CLIP_DURATION)
        
        all_x.append(x)
        all_sr.append(sr)

    return get_stft(all_x)

if __name__ == "__main__":
    recs = split_recording(ENROLL_RECORDING_FNAME)
    print(len(stfts))
    print(stfts[0].shape)
