from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO)

N_CHUNK = 512  # The number of chunks for which the audio dataframe will be developed

GZ = 3000  # The minimum amplitude of the sound.
# If the average amplitude in a chunk is greater than this number,
# then this chunk is assigned the value 1, if not, then 0

PARAM_CH = 5  # The number of consecutive chunks with a value of 1 to which the dataframe will be truncated
# Example:
#   in  [... 0, 1, 0 , 0, 1, 1 ,1, 1, 1, 0, 1, 0, ...]
#   out [1, 1 ,1, 1, 1, 0, 1, 0, ...]

file_list = [f"{dir_name.name}/{file_name.name}"  # list of key, ex: [else/today, ... ]
             for dir_name in os.scandir('audio')
             for file_name in os.scandir(f"audio/{dir_name.name}")]


def break_into_chunks(sp_chunk, n, gz) -> list:
    """Split dataframe into chunks"""
    for chunk in range(len(sp_chunk)):
        res_bool = True
        for a in range(n):
            res_bool *= (abs(np.max(sp_chunk[chunk + a]) - np.min(sp_chunk[chunk + a])) > gz)

        if res_bool:
            return sp_chunk[chunk:]


def clear_empty(input_file: str, output_file: str) -> bool:
    """Remove silence from the file from the beginning to the end"""
    input_data = wav_read(input_file)
    rate = input_data[0]
    data = input_data[1]
    new_data = np.array([[1, 1]])

    sp_chunk = np.split(data, N_CHUNK)
    sp_chunk = break_into_chunks(sp_chunk, PARAM_CH, GZ)
    sp_chunk = break_into_chunks(sp_chunk[::-1], PARAM_CH, GZ)[::-1]

    for i in sp_chunk:
        new_data = np.append(new_data, i, axis=0)

    new_data = new_data[1:]
    wav_write(output_file, rate, new_data.astype(np.int16))

    return True


if __name__ == '__main__':
    start_time = time.time()
    for file in file_list:
        clear_empty(f"audio/{file}", f"audio_clear/{file}")
    logging.info(f"\tProgram execution time: {time.time() - start_time} seconds")
