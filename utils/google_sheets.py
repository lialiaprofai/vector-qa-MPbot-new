import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os # Добавляем импорт os для получения пути к файлам
import logging # Импортируем модуль логирования

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

class GoogleSheetsManager:
    def __init__(self, credentials_path, spreadsheet_id):
        logger.info("Инициализация Google Sheets Manager...")
        full_credentials_path = os.path.abspath(credentials_path)
        logger.info(f"Полный путь к файлу credentials: {full_credentials_path}")

        if not os.path.exists(full_credentials_path):
            logger.error(f"Файл Google credentials не найден по пути: {full_credentials_path}")
            self.credentials = None # Устанавливаем None, чтобы показать ошибку
            self.service = None
            self.spreadsheet_id = None
            return # Прерываем инициализацию при ошибке

        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                full_credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
            self.spreadsheet_id = spreadsheet_id
            logger.info(f"Google Sheets Manager успешно инициализирован для таблицы {self.spreadsheet_id}")
        except Exception as e:
            logger.error(f"Ошибка инициализации Google Sheets Manager: {e}")
            self.credentials = None
            self.service = None
            self.spreadsheet_id = None


    def get_qa_data(self, range_name='Регистрация ТМ!A:D'):
        """
        Чтение данных из Google таблицы.
        Предполагается, что колонка 'Вопрос' - A, 'Ответ' - B, 'Категория' - C, 'Ключевые слова' - D.
        """
        if not self.service:
            logger.error("Google Sheets Service не инициализирован. Не могу прочитать данные.")
            return pd.DataFrame(columns=['Вопрос', 'Ответ', 'Категория', 'Ключевые слова'])

        logger.info(f"Чтение данных из таблицы {self.spreadsheet_id} диапазон {range_name}...")
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            values = result.get('values', [])
            if not values:
                logger.warning(f"Данные из таблицы {self.spreadsheet_id} не получены или таблица пуста.")
                return pd.DataFrame(columns=['Вопрос', 'Ответ', 'Категория', 'Ключевые слова'])

            # Проверяем, что есть хотя бы одна строка (заголовки)
            if len(values) < 1:
                 logger.warning("В таблице нет строк с данными (только заголовки отсутствуют).")
                 return pd.DataFrame(columns=['Вопрос', 'Ответ', 'Категория', 'Ключевые слова'])

            # Предполагаем, что первая строка - это заголовки колонок
            headers = values[0]
            # Проверяем, что заголовки соответствуют ожидаемым (хотя бы по количеству, лучше по именам)
            expected_headers = ['Вопрос', 'Ответ', 'Категория', 'Ключевые слова'] # Ожидаемые заголовки
            if len(headers) < len(expected_headers):
                 logger.warning(f"Количество колонок в таблице ({len(headers)}) меньше ожидаемого ({len(expected_headers)}). Используем первые {len(headers)} заголовков из таблицы.")
                 # Попробуем использовать заголовки из таблицы, но могут быть проблемы
                 pass # Продолжаем с тем, что есть

            # Создаем DataFrame из остальных строк
            # Убедимся, что колонок в данных не больше, чем заголовков
            df = pd.DataFrame(values[1:], columns=headers[:len(values[1]) if values[1] else 0]) # Исправлено для случая пустых строк данных

            # Переименуем колонки, если нужно, или убедимся в их наличии
            # Простая проверка наличия ожидаемых колонок
            for col in expected_headers:
                if col not in df.columns:
                    logger.warning(f"В прочитанных данных отсутствует ожидаемая колонка '{col}'.")
                    # Добавим пустую колонку, чтобы избежать ошибок в дальнейшем коде
                    df[col] = None

            # Оставим только ожидаемые колонки и в правильном порядке
            df = df[expected_headers]


            logger.info(f"Успешно прочитано {len(df)} строк данных из таблицы {self.spreadsheet_id}.")
            return df

        except Exception as e:
            logger.error(f"Ошибка чтения данных из Google Sheets: {e}", exc_info=True) # Логируем ошибку с traceback
            return pd.DataFrame(columns=['Вопрос', 'Ответ', 'Категория', 'Ключевые слова']) # Возвращаем пустой DataFrame в случае ошибки

# Пример использования (можно удалить после тестирования)
# if __name__ == '__main__':
#     # Для запуска примера убедитесь, что у вас установлен GOOGLE_CREDENTIALS и GOOGLE_SHEETS_ID в .env
#     # и установлены библиотеки: pip install google-api-python-client pandas
#     from config import GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID
#     logging.basicConfig(level=logging.INFO) # Устанавливаем уровень логирования для примера

#     if GOOGLE_CREDENTIALS and GOOGLE_SHEETS_ID:
#         logger.info("GOOGLE_CREDENTIALS и GOOGLE_SHEETS_ID загружены.")
#         sheets_manager = GoogleSheetsManager(GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID)
#         if sheets_manager.service: # Проверяем, успешно ли инициализировался сервис
#              qa_data = sheets_manager.get_qa_data()
#              print("\nПрочитанные данные (первые 5 строк):")
#              print(qa_data.head())
#         else:
#              logger.error("Не удалось инициализировать Google Sheets Manager.")

#     else:
#         logger.error("Не удалось загрузить GOOGLE_CREDENTIALS или GOOGLE_SHEETS_ID из config.py")
