import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
# load_dotenv() по умолчанию ищет файл .env в текущей директории
load_dotenv()

# Получаем значения переменных
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_SHEETS_ID = os.getenv('GOOGLE_SHEETS_ID')
GOOGLE_CREDENTIALS = os.getenv('GOOGLE_CREDENTIALS')
MANAGER_CHAT_ID = os.getenv('MANAGER_CHAT_ID')

# Добавим проверку и логирование для удобства
# Вместо print лучше использовать logging.warning или logging.error
# Но print здесь для простоты и быстрого вывода
if not TELEGRAM_TOKEN:
    print("Внимание: Переменная TELEGRAM_TOKEN не установлена в .env")
if not OPENAI_API_KEY:
     print("Внимание: Переменная OPENAI_API_KEY не установлена в .env")
if not GOOGLE_SHEETS_ID:
    print("Внимание: Переменная GOOGLE_SHEETS_ID не установлена в .env")
if not GOOGLE_CREDENTIALS:
     print("Внимание: Переменная GOOGLE_CREDENTIALS не установлена в .env")
if not MANAGER_CHAT_ID:
     print("Внимание: Переменная MANAGER_CHAT_ID не установлена в .env")

# Если хотите прерывать выполнение при отсутствии ключей, раскомментируйте:
# if not all([TELEGRAM_TOKEN, OPENAI_API_KEY, GOOGLE_SHEETS_ID, GOOGLE_CREDENTIALS, MANAGER_CHAT_ID]):
#     raise ValueError("Необходимые переменные окружения не установлены в файле .env")
