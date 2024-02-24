import numpy as np
import wave
import random

def generate_sine_wave(freq, sample_rate, duration):
    # 波形生成のための時間配列
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 正弦波の生成
    y = np.sin(2 * np.pi * freq * t)
    return y

def randomize_wave(data, sample_rate, num_points):
    # データの長さを取得
    length = len(data)

    # ランダムに選択するインデックスのリストを生成
    random_indices = random.sample(range(length), num_points)

    # 選択されたインデックスの値を0に設定
    for index in random_indices:
        data[index] = 0
    
    return data

def save_wave(filename, data, sample_rate):
    # numpy配列を16ビット整数に変換
    data = np.int16(data * 32767)

    # WAVファイルとして保存
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # モノラル
        wav_file.setsampwidth(2)  # サンプル幅（バイト数）：2 = 16ビット
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())

# 正弦波のパラメータ
frequency = 440  # 周波数（Hz）
sample_rate = 44100  # サンプリングレート（Hz）
duration = 10  # 持続時間（秒）

# 正弦波の生成
sine_wave = generate_sine_wave(frequency, sample_rate, duration)

# 正弦波のランダム化
random_sine_wave = randomize_wave(sine_wave.copy(), sample_rate, 100)

# 正弦波の保存
output_file = '/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/sine_wave.wav'
save_wave(output_file, sine_wave, sample_rate)

# ランダム化した正弦波の保存
output_file_random = '/Users/snakano/Desktop/遠隔楽曲制作/music_dvcs_nkn/Dat/random_sine_wave.wav'
save_wave(output_file_random, random_sine_wave, sample_rate)
