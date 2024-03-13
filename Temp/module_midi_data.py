# テストデータ作成用
import sys
sys.path.append('/Users/snakano/Documents/musicxml_analyze')

from library.midi_reader_h import read_midi_file

def modify_note_on_events(track_data):
    modified_data = bytearray()
    i = 0
    while i < len(track_data):
        event = track_data[i]
        if 0x90 <= event <= 0x9F:  # Note On event
            modified_data += track_data[i:i+1]  # Add event type
            modified_data += bytes([track_data[i+1] + 1])  # Modify note number
            modified_data += track_data[i+2:i+3]  # Keep velocity
            i += 3
        else:
            modified_data += track_data[i:i+1]  # Copy other data
            i += 1
    return modified_data



def save_modified_midi(header, tracks, file_path):
    with open(file_path, 'wb') as file:
        # Write header chunk 'MThd' and header size (6 bytes fixed)
        file.write(b'MThd')
        file.write((6).to_bytes(4, byteorder='big'))
        file.write(header['format'].to_bytes(2, byteorder='big')) # 2 bytes for format
        file.write(header['trackcount'].to_bytes(2, byteorder='big')) # 2 bytes for track count
        file.write(header['resolution'].to_bytes(2, byteorder='big')) # 2 bytes for resolution
        # Write tracks
        for track_data in tracks:
            # Each track chunk 'MTrk'
            file.write(b'MTrk')  # Chunk type: MTrk
            modified_track_data = modify_note_on_events(track_data)
            # Write the length of the track data: 4 bytes
            file.write(len(modified_track_data).to_bytes(4, byteorder='big'))
            # Write the modified track data itself
            file.write(modified_track_data)




# 使用例
header, tracks = read_midi_file("Dat/canon.mid")
save_modified_midi(header, tracks, "Dat/modified_file.mid")
