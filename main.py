from IO import midi_importer
from Oracle.midi_oracle import MidiOracle

import mido
from mido import MidiFile


#MIDI データの読み込み
midi_path = "Dat/canon.mid"
midi_file:MidiFile=midi_importer.import_midi(midi_path)
midi_oracle= MidiOracle(midi_file)


