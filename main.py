from IO import midi_importer
from Oracle.midi_oracle import MidiOracle
from MidiAnalyzer.midi_conpartion import *
from MidiAnalyzer.compare_pitch import *
import mido
from mido import MidiFile



#MIDI データの読み込み
midi_path = "Dat/canon.mid"
midi_file:MidiFile=midi_importer.import_midi(midi_path)

#比較テスト用データ
midi_path2="dat/canon2.mid"
midi_file2:MidiFile=midi_importer.import_midi(midi_path2)

#Midiデータを分割して持つインスタンスの作成
midi_oracle= MidiOracle(midi_file)
midi_oracle2= MidiOracle(midi_file2)

#二つのMIDIデータを比較
#compare_midi(midi_oracle,midi_oracle2)

#2つのMIDIデータのイベントを比較（開発途中）
compare_note_events(midi_oracle, midi_oracle2)




