import pyaudio
import wave
import calendar
import os.path
import time
import logging

logging.basicConfig(level=logging.INFO)

N_CHUNK = 1024
CHANNELS = 2
RATE = 44100

RECORD_SECONDS = 2  # Audio file time

time_sleep = 2  # Delay time between different records

months = [i.lower() for i in calendar.month_name if i != '']  # Get months name

words_list = {
    'minute1': 'Минут',
    'minute2': 'Минута',
    'minute3': 'Минуты',
    'hour1': 'Часов',
    'hour2': 'Час',
    'hour3': 'Часа',
    'today': 'Сегодня',
    '2000': 'Две тысячи',
    '22': 'Двадцать второго',
    'year': 'Года',
}


def write_audio(filename: str, rec_seconds) -> bool:
    """Recording an audio file"""
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=N_CHUNK)

    print("* recording")

    frames = []

    for _ in range(0, int(RATE / N_CHUNK * rec_seconds)):
        data = stream.read(N_CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return True


def run_and_voice(path: str, range_voice: iter, rec_seconds: int = RECORD_SECONDS) -> bool:
    """Check for the existence of a file, write it if it doesn't exist"""
    for i in range_voice:
        if not os.path.exists(f"{path}/{i}.wav"):
            if type(range_voice) == dict:
                print(f"\n\n{words_list[i]}\n\n")
            else:
                print(f"\n\n{i}\n\n")
            write_audio(f"{path}/{i}.wav", rec_seconds)
        else:
            continue
    print("Done!!!")
    time.sleep(time_sleep)
    return True


def write_all_files() -> bool:
    """Write down all the necessary files"""
    print("Записываем дни, например: 'первое', 'второе' и т.д.")
    run_and_voice("audio/numbers_day", range(1, 32))

    print("Записываем названия месяцев, например: 'мая', 'ноября' и т.д.")
    run_and_voice("audio/month", months)

    print("Записываем числа минут, например: 'одна', 'двадцать восемь и т.д.")
    run_and_voice("audio/numbers_minute", range(0, 60))

    print("Записываем числа минут, например: 'одна', 'двадцать восемь и т.д.")
    run_and_voice("audio/numbers_hour", range(0, 24))

    print("Записываем различные слова:")
    run_and_voice("audio/else", words_list)
    return True


if __name__ == '__main__':
    start_time = time.time()
    write_all_files()
    logging.info(f"\tProgram execution time: {time.time() - start_time} seconds")
