import mido
from mido import MidiFile
from typing import List, Tuple

"""
MidiFileオブジェクトを受け取ってインスタンス化する
内部で差分計算に必要なデータの解析を行う
"""
class MidiOracle:
    def __init__(self, midi_file: MidiFile):
        # MIDIの読み込み
        self.midi :MidiFile = midi_file
        self.tracks = self.midi.tracks
        #以下の処理で，受け取ったMidiFileのデータを分割して持つ
        

    def analyze_tracks(self) -> List[str]:
        """ トラックを解析する """
        track_info = []
        for track in self.tracks:
            # トラック解析のロジックを実装
            track_info.append(track)
        return track_info

    def get_tempo_changes(self) -> List[Tuple[int, int]]:
        """ テンポ変更イベントを解析し、変更点のリストを返す """
        tempo_changes = []
        # テンポ変更解析のロジックを実装
        return tempo_changes
    
    


if __name__ == "__main__":
    import sys
    sys.path.append('/Users/snakano/Documents/musicxml_analyze')
    from IO import midi_importer

    midi_file = midi_importer.import_midi("Dat/canon.mid")
    Midi_Oracle = MidiOracle(midi_file)
    print(len(Midi_Oracle.analyze_tracks()))#含まれているトラック数

