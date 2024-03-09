import struct

def read_variable_length(file):
    """可変長の値を読み込み、値と読み込んだバイト数を返す"""
    total = 0
    bytes_read = 0
    
    while True:
        byte = file.read(1)
        if not byte:
            break  # ファイルの終わり
        
        byte = ord(byte)
        total = (total << 7) | (byte & 0x7F)
        bytes_read += 1
        if byte & 0x80 == 0:
            break
    
    return total, bytes_read

def read_midi_header(filepath):
    """MIDIファイルのヘッダチャンクを読み込み、解析する"""
    try:
        with open(filepath, 'rb') as file:
            header_data = file.read(14)
    except IOError:
        print(f"Failed to open or read the file: {filepath}")
        return None

    hex_data = header_data.hex().upper()
    header = {
        'chunk_type': hex_data[0:8],
        'header_size': int(hex_data[8:16], 16),
        'format_type': int(hex_data[16:20], 16),
        'tracks': int(hex_data[20:24], 16),
        'division': int(hex_data[24:28], 16)
    }

    return header

def parse_midi_events(file):
    """トラックチャンク内のイベントを解析する"""
    while True:
        delta_time, bytes_read = read_variable_length(file)
        event_type_byte = file.read(1)
        if not event_type_byte:
            break  # ファイルの終わり

        event_type = ord(event_type_byte)
        print(f"Delta Time: {delta_time}, Event Type: {hex(event_type)}")
        
        # ここでイベントタイプに応じた処理を実装
        # 例: ノートオンイベントの処理
        if event_type >= 0x80 and event_type <= 0x8F:
            note_off_channel = event_type & 0xF
            note_number = ord(file.read(1))
            velocity = ord(file.read(1))
            print(f"Note Off - Channel: {note_off_channel}, Note: {note_number}, Velocity: {velocity}")
        # 他のイベントタイプについても同様に処理を追加する

# 使用例
filepath = "./Dat/canon.mid"  # MIDIファイルのパスを適宜設定してください
header_info = read_midi_header(filepath)
if header_info:
    print(header_info)
    with open(filepath, 'rb') as file:
        # ヘッダチャンクをスキップ
        file.read(14)
        # トラックチャンクの処理
        parse_midi_events(file)
else:
    print("MIDI header could not be read.")
