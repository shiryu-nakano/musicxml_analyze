#動かせるようになったらIOに移動する

import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')

import numpy as np

from library import MIDI_file

def import_midi(midi_path):
    """
    指定されたパスからMIDIファイルを読み込み、解析した。
    
    Parameters:
    midi_path (str): 読み込むMIDIファイルのパス

    Returns:
    object: MIDIデータの解析結果
    """
    # MIDI_file.decode は、MIDIファイルを解析して必要なデータを返す仮想の関数です。
    # 実際には、midoやpretty_midiなどのライブラリで適切な処理を行ってください。
    division, tempo, number_of_track, end_of_track, score = MIDI_file.decode(midi_path)
    
    # 解析したMIDIデータを返します。
    # この戻り値はプロジェクトの要件に合わせて適宜調整してください。
    return {
        "division": division,
        "tempo": tempo,
        "number_of_track": number_of_track,
        "end_of_track": end_of_track,
        "score": score
    }

# 使い方の例
# midi_path = 'path/to/your/midi/file.mid'
# midi_data = import_midi(midi_path)

if __name__ == '__main__':
    midi_path="/Users/snakano/Documents/musicxml_analyze/Dat/canon.mid"
    midi=import_midi(midi_path)
    print(type(midi))
    print(midi)



