
import mido
from mido import MidiFile

def import_midi(midi_path:str)-> MidiFile:
    """
    指定されたパスからMIDIファイルを読み込み、Mido.MidiFileオブジェクトを返す
    外部の変更が考えられるのでプログラム内部ではMidiFileとしてのみ扱いたい

    Parameters:
    midi_path (str): 読み込むMIDIファイルのパス

    Returns:
    mido.MidiFile: 読み込んだMIDIファイルのデータ MidiFileのインスタンス
    """
    midi = mido.MidiFile(midi_path)
    return midi



if __name__ == '__main__':
    midi_path="Dat/canon.mid"
    midi=import_midi(midi_path)
    print(type(midi))
    print(midi)