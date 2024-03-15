import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')
from Oracle.midi_oracle import MidiOracle

import mido
from mido import MidiFile

class MidiComparison:
    @staticmethod
    def compare_note_on_events(oracle1: MidiOracle, oracle2: MidiOracle):
        note_events1 = oracle1.get_note_on_events()
        note_events2 = oracle2.get_note_on_events()

        # 音高とタイミングをキーとした辞書に変換
        events_dict1 = {(note, time): track for track, note, time in note_events1}
        events_dict2 = {(note, time): track for track, note, time in note_events2}

        # 音高とタイミングの差分を取得
        diff_keys = set(events_dict1.keys()).symmetric_difference(set(events_dict2.keys()))


        for key in diff_keys:
            note, time = key
            if key in events_dict1:
                print(f"MIDI 1: トラック {events_dict1[key]}、タイミング {time} でノート {note} が存在")
            if key in events_dict2:
                print(f"MIDI 2: トラック {events_dict2[key]}、タイミング {time} でノート {note} が存在")
    
    @staticmethod
    def compare_and_print_events(oracle1, oracle2):
        events1 = oracle1.get_note_on_events()
        events2 = oracle2.get_note_on_events()

        # イベントのマージとソート
        all_events = sorted(events1 + events2, key=lambda x: x[2])  # 絶対時間x[2]に基づいてソート

        # 同じタイミングのイベントを検出し、差異を計算
        current_time = -1
        print("Time: | MIDI1.track1 | MIDI1.track2 | MIDI2.track1 | MIDI2.track2 \n")
        for event in all_events:
            track, note, time = event
            if time != current_time:
                if current_time != -1:
                    print()  # 新しいタイミングの前に改行
                current_time = time
                print(f"Time {time}: ", end="")
            print(f"Track {track} Note {note} | ", end="")

        # ここで差異の出力が必要な場合、さらに複雑なロジックを追加して、
        # 同じタイミングでのイベントの違いを検出し、それを表示します。
        # この例では、単純にイベントを時系列に沿って表示しています。




if __name__ == "__main__":
    from IO import midi_importer
    midi_path1="Dat/canon.mid"
    midi_path2="Dat/canon2.mid"

    midi_file1 = midi_importer.import_midi(midi_path1)
    midi_file2 = midi_importer.import_midi(midi_path2)

    midi_oracle1 = MidiOracle(midi_file1)
    midi_oracle2 = MidiOracle(midi_file2)

    # MIDIデータを比較
    # 2つのMIDIファイル間でノートオンイベントを比較
    #MidiComparison.compare_note_on_events(midi_oracle1,midi_oracle2)
    MidiComparison.compare_and_print_events(midi_oracle1,midi_oracle2)
    