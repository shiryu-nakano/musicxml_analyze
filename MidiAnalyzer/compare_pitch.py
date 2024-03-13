import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')
import mido
from mido import MidiFile
from Oracle.midi_oracle import MidiOracle


"""
開発中．まずは音高が異なる場合に
"""
def compare_note_events(oracle1: MidiOracle, oracle2: MidiOracle):
    
    note_events1 = extract_note_events(oracle1)
    note_events2 = extract_note_events(oracle2)
    
    # タイミングごとにノートオンイベントを比較
    for time in note_events1.keys():
        if time in note_events2:
            notes1 = note_events1[time]
            notes2 = note_events2[time]
            diff_notes = set(notes1).symmetric_difference(set(notes2))
            if diff_notes:
                print(f"タイミング {time} で異なる音高のノートが存在します。差異: {diff_notes}")

def extract_note_events(oracle: MidiOracle) -> dict:
    """MIDIデータを解析する．イベントを検出して格納する

    今後，絶対時間での計算なども拡張予定
    
    """

    note_events = {}
    for track_data in oracle.track_infos:
        for msg in track_data['messages']:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.time not in note_events:
                    note_events[msg.time] = [msg.note]
                else:
                    note_events[msg.time].append(msg.note)
    return note_events



if __name__ == "__main__":
    from IO import midi_importer
    midi_path1="Dat/canon.mid"
    midi_path2="Dat/canon2.mid"

    midi_file1 = midi_importer.import_midi(midi_path1)
    midi_file2 = midi_importer.import_midi(midi_path2)

    midi_oracle1 = MidiOracle(midi_file1)
    midi_oracle2 = MidiOracle(midi_file2)

    # MIDIデータを比較
    #compare_midi(midi_oracle1, midi_oracle2)
    # 2つのMIDIファイル間でノートオンイベントを比較
    compare_note_events(midi_oracle1, midi_oracle2)
