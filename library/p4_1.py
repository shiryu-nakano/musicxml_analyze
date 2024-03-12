import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')

import numpy as np
from wave_file import wave_write_16bit_mono
from MidiAnalyzer import MIDI_file
from MidiAnalyzer.MIDI_file import decode

def sine_wave(fs, note_number, velocity, gate):
    length_of_s = int(fs * gate)
    s = np.zeros(length_of_s)
    f = 440 * np.power(2, (note_number - 69) / 12)
    for n in range(length_of_s):
        s[n] = np.sin(2 * np.pi * f * n / fs)

    for n in range(int(fs * 0.01)):
        s[n] *= n / (fs * 0.01)
        s[length_of_s - n - 1] *= n / (fs * 0.01)

    gain = velocity / 127 / np.max(np.abs(s))
    s *= gain
    return s

division, tempo, number_of_track, end_of_track, score = decode('Dat/canon.mid')
tempo = 60 / (tempo / 1000000)
number_of_track = int(number_of_track - 1)
end_of_track = (end_of_track / division) * (60 / tempo)
number_of_note = score.shape[0]

fs = 44100
length_of_s_master = int(fs * (end_of_track + 2))
track = np.zeros((length_of_s_master, number_of_track))
s_master = np.zeros(length_of_s_master)

for i in range(number_of_note):
    j = int(score[i, 0] - 1)
    onset = (score[i, 1] / division) * (60 / tempo)
    note_number = score[i, 2]
    velocity = score[i, 3]
    gate = (score[i, 4] / division) * (60 / tempo)
    s = sine_wave(fs, note_number, velocity, gate)
    length_of_s = len(s)
    offset = int(fs * onset)
    for n in range(length_of_s):
        track[offset + n, j] += s[n]

for j in range(number_of_track):
    for n in range(length_of_s_master):
        s_master[n] += track[n, j]

master_volume = 0.5
max_abs_value = np.max(np.abs(s_master))
if max_abs_value == 0:
    print("Maximum value is zero, cannot normalize.")
    s_master = np.zeros_like(s_master)  # Create an array of zeros if max is 0 to avoid division by zero
else:
    s_master /= max_abs_value  
s_master *= master_volume

#wave_write_16bit_mono(fs, s_master.copy(), 'Temp/p4_1(output)2.wav')
wave_write_16bit_mono(fs, s_master, 'Temp/p4_1(output)2.wav')