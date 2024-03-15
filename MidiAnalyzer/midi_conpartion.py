import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')
import mido
from mido import MidiFile
from Oracle.midi_oracle import MidiOracle

#こっちはまだ動きません！
#実行可能ですが，想定した出力結果は得られません

def compare_tracks(track1: dict, track2: dict):
    # イベントの種類ごとに比較
    for event_type in ['messages', 'meta_messages', 'sysex_messages']:
        events1 = track1[event_type]
        events2 = track2[event_type]

        # 特定のイベントタイプ（ノートオン、ノートオフ、コントロールチェンジ）に絞る
        if event_type == 'messages':
            note_ons1 = [msg for msg in events1 if msg.type == 'note_on' and msg.velocity > 0]
            note_offs1 = [msg for msg in events1 if (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))]
            control_changes1 = [msg for msg in events1 if msg.type == 'control_change']

            note_ons2 = [msg for msg in events2 if msg.type == 'note_on' and msg.velocity > 0]
            note_offs2 = [msg for msg in events2 if (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))]
            control_changes2 = [msg for msg in events2 if msg.type == 'control_change']

            # 例: ノートオンイベントの比較
            if len(note_ons1) != len(note_ons2):
                print(f"ノートオンイベント数が異なります。トラック1: {len(note_ons1)}, トラック2: {len(note_ons2)}")
            # さらに詳細な比較が必要な場合はここに追記

def compare_midi(midi1: MidiOracle, midi2: MidiOracle):
    for i, (track1, track2) in enumerate(zip(midi1.track_infos, midi2.track_infos)):
        print(f"トラック {i} の比較:")
        compare_tracks(track1, track2)

def analyze_note_height_differences(midi_oracle: MidiOracle):
    for track_index, track_data in enumerate(midi_oracle.track_infos):
        # 各トラックのノートオンイベントを保持する辞書
        note_on_events = {}
        
        for msg in track_data['messages']:
            if msg.type == 'note_on' and msg.velocity > 0:  # ノートオンイベントとして扱う
                if msg.time not in note_on_events:
                    note_on_events[msg.time] = [msg.note]
                else:
                    note_on_events[msg.time].append(msg.note)

        # 同じタイミングで異なる音高が鳴っているイベントを探す
        for time, notes in note_on_events.items():
            if len(notes) > 1:  # 同じタイミングで複数の音が鳴っている
                print(f"トラック {track_index}、タイミング {time}: 異なる音高のノート {notes}")


       


if __name__ == "__main__":
    from IO import midi_importer
    midi_path1="Dat/canon.mid"
    midi_path2="Dat/canon2.mid"

    midi_file1 = midi_importer.import_midi(midi_path1)
    midi_file2 = midi_importer.import_midi(midi_path2)

    midi_oracle1 = MidiOracle(midi_file1)
    midi_oracle2 = MidiOracle(midi_file2)

    # MIDIデータを比較
    compare_midi(midi_oracle1, midi_oracle2)
