import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')
import mido
from mido import MidiFile, MidiTrack, Message
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

    def analyze_tracks(self) -> List[List[Message]]:
        """トラックを解析し、各トラックのMIDIメッセージのリストを返す"""
        track_infos = []
        for track in self.tracks:
            track_msgs = [msg for msg in track if isinstance(msg, Message)]
            track_infos.append(track_msgs)
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
    # MidiOracle クラスの定義を前提とします

    # 使用するMIDIファイルを指定
    midi_file_path = 'Dat/canon.mid'

    # Midoを使ってMIDIファイルを読み込む
    midi_file = MidiFile(midi_file_path)

    # MidiOracle クラスのインスタンスを作成
    oracle = MidiOracle(midi_file)
    print("Imported MIDI file:",midi_file_path)
    # トラック情報の取得と出力
    print("トラック情報:")
    for i, track_msgs in enumerate(oracle.track_infos):
        print(f"トラック {i}: メッセージ数 {len(track_msgs)}")
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
