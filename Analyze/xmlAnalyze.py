"""
二つのWavファイルを読み込み、その内部の異なる部分を抽出する
"""


import wave
import numpy as np

def read_wav(file_path):
    with wave.open(file_path, 'r') as wav_file:
        # パラメータ取得
        nchannels, sampwidth, framerate, nframes, comptype, compname = wav_file.getparams()

        # モノラルチェック
        if nchannels != 1:
            raise ValueError("ファイルはモノラルでなければなりません。")

        # データ読み込み
        frames = wav_file.readframes(nframes)
        wav_data = np.frombuffer(frames, dtype=np.int16)
        
        return wav_data

def compare_waves(file1, file2):
    data1 = read_wav(file1)
    data2 = read_wav(file2)

    # 長さが異なる場合は短い方に合わせる
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # 差異を検出
    differences = np.where(data1 != data2)[0]
    difference_count:int =0

    # 差異のある箇所とその前後の値を表示
    for index in differences:
        start = max(0, index - 1)
        end = min(len(data1), index + 2)
        difference_count=difference_count+1
        print(f"Index: {index}, Values in File 1: {data1[start:end]}, Values in File 2: {data2[start:end]}")
    
    print("differences: ",difference_count)



# 使用例
file1 = r'/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/random_sine_wave.wav'
file2 = r'/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/sine_wave.wav'

compare_waves(file1, file2)
