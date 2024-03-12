

def get_int_from_bytes(byte_array):
    """バイト配列から整数を計算するヘルパー関数"""
    return int.from_bytes(byte_array, byteorder='big')

def parse_header(chunk_data):
    """ヘッダチャンクの解析"""
    if chunk_data[0:4] != b'MThd':
        raise ValueError("Invalid SMF header chunk")
    header_size = get_int_from_bytes(chunk_data[4:8])
    format_type = get_int_from_bytes(chunk_data[8:10])
    track_count = get_int_from_bytes(chunk_data[10:12])
    resolution = get_int_from_bytes(chunk_data[12:14])
    return {
        'size': header_size,
        'format': format_type,
        'trackcount': track_count,
        'resolution': resolution
    }


def parse_tracks(data, start_offset):
    """トラックチャンクの解析"""
    tracks = []
    offset = start_offset
    while offset < len(data):
        if data[offset:offset+4] != bytes([0x4D, 0x54, 0x72, 0x6B]):
            raise ValueError("Invalid track chunk")
        track_size = get_int_from_bytes(data[offset+4:offset+8])
        track_data = data[offset+8:offset+8+track_size]
        tracks.append(track_data)
        offset += 8 + track_size
    return tracks

def read_midi_file(file_path):
    """MIDIファイルを読み込み、ヘッダとトラックを解析する"""
    with open(file_path, 'rb') as file:
        data = file.read()

    header = parse_header(data[:14])
    tracks = parse_tracks(data, 14)
    
    return header, tracks


#可視化による構造理解のため
def decode_midi_event(event_byte):
    if event_byte == 0xFF:
        return "Meta Event"
    elif event_byte >= 0x80 and event_byte <= 0x8F:
        return "Note Off"
    elif event_byte >= 0x90 and event_byte <= 0x9F:
        return "Note On"
    elif event_byte >= 0xA0 and event_byte <= 0xAF:
        return "Polyphonic Key Pressure"
    elif event_byte >= 0xB0 and event_byte <= 0xBF:
        return "Control Change"
    elif event_byte >= 0xC0 and event_byte <= 0xCF:
        return "Program Change"
    elif event_byte >= 0xD0 and event_byte <= 0xDF:
        return "Channel Pressure"
    elif event_byte >= 0xE0 and event_byte <= 0xEF:
        return "Pitch Bend Change"
    else:
        return "Unknown Event"

def parse_track_events(track_data):
    i = 0
    events = []
    while i < len(track_data):
        if track_data[i] == 0xFF:  # Meta event
            meta_type = track_data[i+1]
            length = track_data[i+2]
            description = f"Meta Event: Type {meta_type}, Length {length}"
            events.append(description)
            i += 3 + length  # Skip to next event
        elif 0x80 <= track_data[i] <= 0xEF:  # MIDI event
            event_type = decode_midi_event(track_data[i])
            description = f"MIDI Event: {event_type}"
            # Note On/Off events have 2 additional bytes (note and velocity)
            if event_type == "Note On" or event_type == "Note Off":
                note = track_data[i+1]
                velocity = track_data[i+2]
                description += f", Note {note}, Velocity {velocity}"
                i += 3
            else:
                i += 2  # Other MIDI events
            events.append(description)
        else:
            i += 1  # Skip unknown or unhandled bytes
    return events



if __name__ =="__main__":
    header, tracks=read_midi_file("Dat/canon.mid")

    print(type(header))
    print(header,"\n")
    
    print(type(tracks))
    print(tracks)
    # Example usage with your provided track data

    # track の格納状況を見る
    print("Length of \"tarck\" :",len(tracks))

    """
    for idx, track_data in enumerate(tracks): print(f"Track {idx+1}:")
       events = parse_track_events(track_data)
       for event in events:
           print(event)
       print("---------")
    """
    # 修正後のMIDIファイルを読み込んでヘッダ情報を表示する
    #modified_header, modified_tracks = read_midi_file("Dat/canon2.mid")
    #print(modified_header)
