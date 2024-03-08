
import mido
from mido import MidiFile

def import_midi(midi_path:str)-> MidiFile:
    """
    指定されたパスからMIDIファイルを読み込み、Mido MidiFileオブジェクトを返す
    
    Parameters:
    midi_path (str): 読み込むMIDIファイルのパス

    Returns:
    mido.MidiFile: 読み込まれたMIDIファイルのデータ
    """
    midi = mido.MidiFile(midi_path)
    return midi



if __name__ == '__main__':
    midi_path="/Users/snakano/Documents/musicxml_analyze/Dat/canon.mid"
    midi=import_midi(midi_path)
    print(type(midi))
    print(midi)

# for i, track in enumerate(midi.tracks):
#     print('Track {}: {}'.format(i, track.name))
#     for msg in track:
#         print(msg)