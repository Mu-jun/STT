import numpy as np
import speech_recognition as sp_rec

from model import ServiceModel

sr = 16000
ch = 1

rec = sp_rec.Recognizer()
mic = sp_rec.Microphone(sample_rate=sr)
model = ServiceModel()

def service(sig):
    print(model.one_shot(np.array(sig).flatten()))

global running
running= True

try:
    with mic:
        procs = []
        print('start running')
        while running:
            sig = rec.listen(mic).get_wav_data(sr,2)
            data_s16 = np.frombuffer(sig, dtype=np.int16, count=len(sig)//2)
            float_data = (data_s16 * 0.5**15).astype(np.float32)
            service(float_data)
except KeyboardInterrupt as ke:
    print("finished")
except Exception as e:
    print(e)
    
for p in procs:
    p.join()
    
print("Program end")