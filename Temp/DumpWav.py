import os
import wave
import numpy as np


def process_wav_file(file_path):
    with wave.open(file_path, 'r') as wav_file:
        params = wav_file.getparams()
        print(f"File: {file_path}")
        print(f"Channels: {params.nchannels}")
        print(f"Sample Width: {params.sampwidth}")
        print(f"Frame Rate: {params.framerate}")
        print(f"Number of Frames: {params.nframes}")
        print(f"Compression Type: {params.comptype}")
        print(f"Compression Name: {params.compname}\n")

# 指定されたディレクトリへのパス
directory_path = r'/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/Brit_Clean13.wav'

# ディレクトリ内のすべてのファイルをループ処理
"""
for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory_path, filename)
        process_wav_file(file_path)
"""

process_wav_file(directory_path)


#　Wav モジュールを用いた　wav ファイルのダンプ
def DumpWave(filename:str):
    wav=wave.open(filename,'r')
    nachannels, sampwith, framerate, nframes, comptype, compname= wav.getparams()

    data= wav.readframes(nframes)
    wav_data=np.frombuffer(data,dtype='int16')

    for n in wav_data: 
        print(n)

    wav.close
    print(nachannels)


# dumping wav file by using soundfile module
def DumpfilebySoundfile(filename:str):
    import soundfile as sf
    data, fs = sf.read(filename)
    data = data.flatten()
    for n in data:
        print(n)
        



if __name__=="__main__":
    # 指定されたディレクトリへのパス
    directory_path = r'/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/Brit_Clean13.wav'
    DumpWave(directory_path)
    #DumpfilebySoundfile(directory_path)
    
