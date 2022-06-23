import numpy as np
import librosa
import sounddevice as sd
from queue import Queue
import time
import sys

from model import ServiceModel

sr = 16000
ch = 1

global running, threshold
running = True
sig_flag = False

q = Queue()

def rms(array):
    return np.sqrt( np.mean( np.array(array) **2 ) )

def callback(indata, frames, time, status):
    global threshold, sig_flag, cnt, pause_cnt
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def hpss(array):
    if len(array) > 2048:
        stft = librosa.stft(array)
        hamonic, _ = librosa.decompose.hpss(stft)
        return librosa.istft(hamonic)
    else:
        return None
    
def get_q():
    array = []
    while not q.empty():
        array.extend(q.get())
    if len(array) > 1:
        return np.concatenate(array)
    else:
        return np.array(array)
    
model = ServiceModel()

def service(sig):
    print(model.one_shot(np.array(sig).flatten()))
    
    
if __name__ == "__main__":
    pause_cnt = 10
    sig_flag = False
    speech_sig = np.zeros(1,dtype=np.float32)
    try:
        print("init start - please be quiet")
        with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
            time.sleep(2)
        sig = get_q()
        print(sig.dtype)
        sig = hpss(sig)
        threshold = np.max(np.abs(sig))
        print("init complete")
        print("threshold :",threshold)
        with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
            sig = np.zeros(1,dtype=np.float32)
            print(sig.dtype)
            print("running start")
            
            while running:
                # time.sleep(0.05)
                sig = np.concatenate((sig, get_q()))
                harmonic_sig = hpss(sig)
                
                if harmonic_sig is None:
                    continue
                
                if rms(harmonic_sig) > threshold:
                    print("saying")
                    # time.sleep(1)
                    speech_sig = np.concatenate((speech_sig,sig))
                    sig_flag = True
                    cnt = 0
                else:
                    if sig_flag:
                        if cnt < pause_cnt:
                            speech_sig = np.concatenate((speech_sig,sig))
                            cnt += 1
                        else:
                            print("STT")
                            service(speech_sig)
                            speech_sig = np.zeros(1,dtype=np.float32)
                            sig_flag = False

                sig = np.zeros(1,dtype=np.float32)
                
    except KeyboardInterrupt as ke:
        print("Recording finished")
        
    print("Program end")