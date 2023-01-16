## Sounding_Time
###### Zyblyuk Roman


# Запуск Проекта 
Создайте папку json
Создайте папки audio и audio_clear
а в них папки else, month, numbers_day, numbers_hour, numbers_minute

Создайте виртуальную среду Python и установите зависимости:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

После установки зависимостей запишите голос запустив скрипт create_audio_files.py: 

```
python create_audio_files.py
```

Затем нужно очистить звук от лишней тишины скриптом clear_empty.py: 
```
python clear_empty.py
```

Запись в json: 
```
python to_json.py
```

Приготовления закончены, теперь можно узнать время запустив main.py
```
python main.py
```
