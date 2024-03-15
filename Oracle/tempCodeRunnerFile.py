
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
