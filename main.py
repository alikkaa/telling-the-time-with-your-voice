import datetime
import calendar
import time
import simpleaudio as sa
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


month = dict([
    (i, name.lower())
    for i, name in zip(range(0, 13), calendar.month_name)
    if name != ''])  # creating a dictionary of months, ex: {1: january, ...}

data = pd.read_json("json/recorded_voice.json").to_dict()  # read json file


def get_sound(key: str) -> bool:
    """Get sound from json file"""
    rate = data[key]['Rate']
    chanel1 = data[key]['Channel1']
    chanel2 = data[key]['Channel2']

    chanel1 = bytearray.fromhex(chanel1)
    chanel2 = bytearray.fromhex(chanel2)

    chanel1 = np.frombuffer(chanel1, np.int16)
    chanel2 = np.frombuffer(chanel2, np.int16)

    audio = np.array([chanel1, chanel2])
    audio = np.transpose(audio, axes=None)

    audio = audio.copy(order='C')

    play_obj = sa.play_buffer(audio, 2, 2, rate)
    play_obj.wait_done()
    return True


def get_datetime_audio() -> bool:
    """Get the current time and play the corresponding sound"""
    def get_h_m_type(num: int, h_m: str):
        if num % 10 in range(2, 5) and num not in range(12, 15):
            get_sound(f"else/{h_m}3.wav")
        elif num % 10 == 1 and num != 11:
            get_sound(f"else/{h_m}2.wav")
        else:
            get_sound(f"else/{h_m}1.wav")

    now = datetime.datetime.now()
    get_sound(f"else/today.wav")
    get_sound(f"numbers_day/{now.day}.wav")
    get_sound(f"month/{month[now.month]}.wav")
    get_sound(f"numbers_hour/{now.hour}.wav")

    get_h_m_type(now.hour, "hour")

    get_sound(f"numbers_minute/{now.minute}.wav")

    get_h_m_type(now.minute, "minute")

    get_sound(f"else/2000.wav")
    get_sound(f"else/22.wav")
    get_sound(f"else/year.wav")
    return True


if __name__ == '__main__':
    start_time = time.time()
    get_datetime_audio()
    logging.info(f"\tProgram execution time: {time.time() - start_time} seconds")
