from scipy.io.wavfile import read as wav_read
import pandas as pd
import json
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

audio_dir = 'audio_clear'  # directory of recorded voices

file_list = [f"{dir_name.name}/{file_name.name}"  # list of key, ex: [else/today, ... ]
             for dir_name in os.scandir(audio_dir)
             for file_name in os.scandir(f"{audio_dir}/{dir_name.name}")]


def to_json(filelist: list, output_scv: str) -> bool:
    """Put all recorded audio files into one json"""
    df = pd.DataFrame(columns=['Key', 'Rate', 'Channel1', 'Channel2'])

    for filename in filelist:
        input_data = wav_read(f"{audio_dir}/{filename}")
        word = filename
        df.loc[len(df.index)] = [word, input_data[0],
                                 input_data[1][:, 0].tobytes().hex(),
                                 input_data[1][:, 1].tobytes().hex()]

    with open(output_scv, "w") as f:
        json.dump(df.set_index('Key').to_dict('index'), f)
    return True


if __name__ == "__main__":
    start_time = time.time()
    to_json(file_list, f"json/recorded_voice.json")
    logging.info(f"\tProgram execution time: {time.time() - start_time} seconds")
