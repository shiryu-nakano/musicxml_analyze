import os
import wave
import numpy as np
import matplotlib.pyplot as plt  


#applied to Stereo 
def dispWav(filename:str):
    in_wav = wave.Wave_read(filename)
    data = in_wav.readframes(in_wav.getnframes())
    data = np.frombuffer(data, dtype='int16')

    #reshape 
    nchannels = in_wav.getnchannels()
    if nchannels ==2 :
        data = data.reshape(data.size//2, nchannels) 
    else :
        data
    print(nchannels)
    print(np.shape(data))

    # 波形表示
    t = np.arange(0,data.size/in_wav.getframerate(), 1/in_wav.getframerate())
    print(np.shape(t))
    
    plt.plot(data)
    plt.title("draw wave")
    plt.grid(True)
    plt.show()



if __name__=="__main__":
    # 指定されたディレクトリへのパス
    directory_path = r'/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/Brit_Clean13.wav'
    dispWav(directory_path)
