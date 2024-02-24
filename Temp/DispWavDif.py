import wave
import numpy as np
import matplotlib.pyplot as plt

def read_wav(file_path):
    with wave.open(file_path, 'r') as wav_file:
        nchannels, sampwidth, framerate, nframes, comptype, compname = wav_file.getparams()
        if nchannels != 1:
            raise ValueError("ファイルはモノラルでなければなりません。")
        frames = wav_file.readframes(nframes)
        wav_data = np.frombuffer(frames, dtype=np.int16)
        return wav_data

def plot_waves(file1, file2):
    data1 = read_wav(file1)
    data2 = read_wav(file2)

    # 最短のデータ長に合わせる
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # 波形をプロット
    plt.figure(figsize=(10, 4))
    plt.plot(data1, label='ファイル1', color='blue')
    plt.plot(data2, label='ファイル2', color='red', alpha=0.7)  # alphaで透明度を調整
    plt.title('WAVファイルの波形比較')
    plt.xlabel('サンプル')
    plt.ylabel('振幅')
    plt.legend()
    plt.show()

# 使用例
file1 = '/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/sine_wave.wav'
file2 = '/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/random_sine_wave.wav'
plot_waves(file1, file2)
