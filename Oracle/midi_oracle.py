import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple


class MidiOracle:
    """
    MidiOracleクラス

    Note:
        受け取ったMidiFileの情報を分割して持つ
    Attributes
        midi (MidiFile): MidiFileオブジェクト
        ticks_per_beat (int): 1拍あたりのTick数
        tracks (List[MidiTrack]): MidiFileオブジェクトから取得したトラックのリスト
        track_infos (List[List[Message]]): 各トラックのMIDIメッセージのリスト
        tempo_changes (List[Tuple[int, int]]): テンポ変更イベントのリスト (tick, tempo)
    """
    def __init__(self, midi_file: MidiFile):
        self.midi = midi_file
        self.ticks_per_beat = midi_file.ticks_per_beat  # タイムディビジョンの解像度
        self.tracks = self.midi.tracks
        self.track_infos = self.analyze_tracks()
        self.tempo_changes = self.get_tempo_changes()

    def analyze_tracks(self) -> List[dict[str, List[Message]]]:
        """トラックを解析し、各トラックのMIDIメッセージとメタイベントのリストを返す"""
        track_infos = []
        for track in self.tracks:
            track_data = {
                'messages': [],
                'meta_messages': [],
                'sysex_messages': []
            }
            for msg in track:
                if isinstance(msg, Message):
                    if msg.type == 'sysex':
                        track_data['sysex_messages'].append(msg)
                    else:
                        track_data['messages'].append(msg)
                elif isinstance(msg, MetaMessage):
                    track_data['meta_messages'].append(msg)
            track_infos.append(track_data)
        return track_infos

    def get_tempo_changes(self) -> List[Tuple[int, int]]:
        """テンポ変更イベントを解析し、(tick, tempo)のリストを返す"""
        tempo_changes = []
        for track in self.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo_changes.append((msg.time, msg.tempo))
        return tempo_changes

    def get_track_names(self) -> List[str]:
        """各トラックの名前を取得"""
        track_names = []
        for track in self.tracks:
            name = ''
            for msg in track:
                if msg.type == 'track_name':
                    name = msg.name
                    break
            track_names.append(name)
        return track_names



if __name__ =="__main__":



    midi_file_path = 'Dat/canon.mid'

    # Midoを使ってMIDIファイルを読み込む
    midi_file = MidiFile(midi_file_path)

    # MidiOracle クラスのインスタンスを作成
    oracle = MidiOracle(midi_file)
    print("Imported MIDI file:",midi_file_path)

    # トラック情報の取得と出力
    print("トラック情報:")
    for i, track_data in enumerate(oracle.track_infos):
        print(f"\nトラック {i}: メッセージ数 {len(track_data)}")
        for msg in track_data['messages']:
            print(f"    {msg}")
        print(f"  メタメッセージ数: {len(track_data['meta_messages'])}")
        for meta_msg in track_data['meta_messages']:
            print(f"    {meta_msg}")
        print(f"  SysExメッセージ数: {len(track_data['sysex_messages'])}")
        for sysex_msg in track_data['sysex_messages']:
            print(f"    {sysex_msg}")
 
    print("\n")

    # テンポ変更イベントの取得と出力
    print("テンポ変更イベント:")
    for tempo_change in oracle.tempo_changes:
        print(f"Tick: {tempo_change[0]}, Tempo: {tempo_change[1]}")
    print("\n")

    # トラック名の取得と出力
    print("トラック名:")
    for name in oracle.get_track_names():
        print(name)
