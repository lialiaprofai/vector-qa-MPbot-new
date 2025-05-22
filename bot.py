import os
import logging
import requests # Для отправки в Make.com
import telegram # Импортируем библиотеку telegram полностью для send_message
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from dotenv import load_dotenv

# Импортируем классы, которые мы создали
from config import TELEGRAM_TOKEN, MANAGER_CHAT_ID, GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID, OPENAI_API_KEY
from utils.google_sheets import GoogleSheetsManager
from database.vector_store import VectorStore

# --- Настройка логирования ---
# Устанавливаем базовый уровень логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Получаем логгер для нашего бота
logger = logging.getLogger(__name__)

# --- Класс Бота ---
class QABot:
    def __init__(self):
        # Загружаем переменные окружения из .env
        load_dotenv()

        # Проверяем наличие всех необходимых переменных
        if not all([TELEGRAM_TOKEN, MANAGER_CHAT_ID, GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID, OPENAI_API_KEY]):
             logger.error("Некоторые необходимые переменные окружения не установлены. Проверьте файл .env")
             # В реальном приложении здесь можно вызвать исключение или завершить работу
             # Для простоты пока просто выведем сообщение
             # raise ValueError("Необходимые переменные окружения не установлены")

        # Инициализация менеджера Google Sheets
        # Убедимся, что путь к файлу credentials.json корректен относительно запуска скрипта
        # os.path.join() объединяет путь к текущей директории скрипта с относительным путем из .env
        # Или просто используйте абсолютный путь в .env, если так удобнее
        # Если в .env путь уже относительный от корня проекта (./), то он будет корректно передан
        self.sheets_manager = GoogleSheetsManager(GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID)

        # Инициализация векторной базы данных
        # База данных будет сохранена в папке 'db' в корне проекта
        self.vector_store = VectorStore(db_path="./db")

        # Загрузка данных из Google Sheets в векторную базу при запуске
        self.load_qa_data()

        logger.info("Бот инициализирован и готов к работе.")

    def load_qa_data(self):
        """
        Загружает вопросы и ответы из Google Sheets в векторную базу данных.
        """
        logger.info("Начало загрузки данных из Google Sheets...")
        qa_data_df = self.sheets_manager.get_qa_data()

        if not qa_data_df.empty:
            logger.info(f"Прочитано {len(qa_data_df)} строк из Google Sheets. Начинаем добавление в векторную базу.")
            # Можно добавить прогресс-бар для больших таблиц (с помощью tqdm)
            for index, row in qa_data_df.iterrows():
                question = row.get('Вопрос')
                answer = row.get('Ответ')
                category = row.get('Категория', 'general') # Добавляем категорию, если есть
                # Ключевые слова можно использовать для метаданных или как часть текста для эмбеддинга
                # keywords = row.get('Ключевые слова', '')

                if question and answer: # Проверяем, что вопрос и ответ не пустые
                    metadata = {'category': category} # Можно добавить другие метаданные
                    self.vector_store.add_qa_pair(question, answer, metadata=metadata)
                else:
                     logger.warning(f"Пропущена строка {index + 2} из-за отсутствия вопроса или ответа: {row.to_dict()}") # Логируем пропущенные строки

            logger.info(f"Загрузка данных завершена. В векторной базе {self.vector_store.count()} элементов.")
        else:
            logger.warning("Не удалось загрузить данные из Google Sheets или таблица пуста.")


    # --- Обработчики команд Telegram ---
    def start_command(self, update: Update, context: CallbackContext) -> None:
        """Обработчик команды /start."""
        user = update.effective_user
        logger.info(f"Получена команда /start от пользователя {user.id}")
        update.message.reply_html(
            f"Привет, {user.mention_html()}! Я бот-помощник. Задайте мне ваш вопрос.",
            # reply_markup=ForceReply(selective=True), # Можно добавить, чтобы бот ждал ответа
        )

    def help_command(self, update: Update, context: CallbackContext) -> None:
         """Обработчик команды /help."""
         logger.info(f"Получена команда /help от пользователя {update.effective_user.id}")
         update.message.reply_text("Задайте мне вопрос о продукте или сервисе, и я постараюсь найти ответ. Если ответа нет, я передам ваш вопрос менеджеру.")

    # --- Обработчик текстовых сообщений ---
    def handle_message(self, update: Update, context: CallbackContext) -> None:
        """Обрабатывает входящие текстовые сообщения."""
        user_message = update.message.text
        user_id = update.effective_user.id
        logger.info(f"Получено сообщение от пользователя {user_id}: {user_message}")

        # Поиск в векторной базе
        # Ищем N наиболее похожих результатов
        search_results = self.vector_store.search_similar(user_message, n_results=1) # Ищем только 1 наиболее похожий результат

        # Проверяем результаты поиска
        found_answer = None
        # Проверяем дистанцию найденного результата, чтобы отфильтровать нерелевантные ответы
        # Значение порога (threshold) нужно подбирать экспериментально. Чем меньше значение, тем точнее совпадение.
        # Например, если дистанция > 0.2 (или 0.3), считаем, что ответ не найден.
        distance_threshold = 0.3 # Примерное пороговое значение дистанции
        if search_results and search_results.get('documents') and search_results.get('distances'):
            # Проверяем дистанцию первого (наиболее похожего) результата
            if search_results['distances'][0] <= distance_threshold:
                found_answer = search_results['documents'][0]
                logger.info(f"Найден релевантный ответ в базе для пользователя {user_id} (дистанция: {search_results['distances'][0]:.4f}).")
            else:
                 logger.info(f"Найден ответ с низкой релевантностью (дистанция: {search_results['distances'][0]:.4f}) для пользователя {user_id}. Считаем, что ответ не найден.")


        if found_answer:
            # Отправляем найденный ответ пользователю
            update.message.reply_text(found_answer)
        else:
            # Если ответ не найден или нерелевантен, отправляем вопрос менеджеру
            logger.info(f"Ответ не найден или нерелевантен в базе для пользователя {user_id}. Передаем вопрос менеджеру.")
            self.send_to_manager(user_message, user_id, update.effective_user.full_name) # Передаем имя пользователя
            update.message.reply_text("Спасибо за ваш вопрос! Я не нашёл ответ в базе или найденный ответ нерелевантен, передал его менеджеру, и мы скоро с вами свяжемся.")


    # --- Функция отправки сообщения менеджеру (через Make.com) ---
    def send_to_manager(self, question: str, user_id: int, user_name: str):
        """
        Отправляет вопрос менеджеру, например, через Telegram API напрямую
        или через Webhook Make.com.

        Эта функция пока является заглушкой. Вам нужно будет реализовать логику отправки.
        """
        logger.info(f"Попытка отправки вопроса менеджеру от {user_name} (ID: {user_id}): {question}")

        # --- Вариант 1: Отправка напрямую в чат менеджеров через Telegram API ---
        # Это самый простой способ. Использует токен бота для отправки сообщения.
        # Убедитесь, что MANAGER_CHAT_ID установлен правильно (с минусом для групп)
        if TELEGRAM_TOKEN and MANAGER_CHAT_ID:
             try:
                 # Используем telegram.Bot напрямую
                 # Создаем экземпляр бота с загруженным токеном
                 bot_instance = telegram.Bot(token=TELEGRAM_TOKEN)
                 message_text = f"❗️ **Новый вопрос от пользователя** ❗️\n\n**От:** {user_name} (ID: {user_id})\n**Вопрос:** {question}"
                 bot_instance.send_message(chat_id=MANAGER_CHAT_ID, text=message_text, parse_mode='Markdown') # Используем Markdown для форматирования
                 logger.info(f"Сообщение с вопросом отправлено менеджеру в чат {MANAGER_CHAT_ID}")
             except Exception as e:
                 logger.error(f"Ошибка при отправке сообщения менеджеру через Telegram API: {e}")
                 # Можно отправить сообщение пользователю, что не удалось связаться с менеджером
                 # update.message.reply_text("К сожалению, произошла ошибка при передаче вашего вопроса менеджеру.")
        else:
             logger.warning("Не установлен TELEGRAM_TOKEN или MANAGER_CHAT_ID. Не могу отправить сообщение менеджеру.")


        # --- Вариант 2 (Заглушка для Make.com): Отправка через Webhook Make.com ---
        # Если вы планируете использовать Make.com для более сложной логики (например,
        # сохранения в Google Sheets, отправки email и т.д.),
        # то здесь нужно отправить HTTP POST запрос на webhook Make.com.

        # Пример (ЗАГЛУШКА, замените на реальный URL вашего webhook Make.com):
        # MAKE_WEBHOOK_URL = "ВАШ_URL_WEBHOOK_MAKE"
        # import time # Добавить import time в начале файла если используете
        # if MAKE_WEBHOOK_URL:
        #     payload = {
        #         "user_id": user_id,
        #         "user_name": user_name,
        #         "question": question,
        #         "timestamp": int(time.time()) # Добавляем метку времени
        #     }
        #     try:
        #         response = requests.post(MAKE_WEBHOOK_URL, json=payload)
        #         response.raise_for_status() # Вызовет исключение для ошибок HTTP
        #         logger.info(f"Вопрос отправлен в Make.com Webhook. Статус: {response.status_code}")
        #     except requests.exceptions.RequestException as e:
        #         logger.error(f"Ошибка при отправке вопроса в Make.com Webhook: {e}")
        # else:
        #     logger.warning("Не установлен URL Webhook Make.com. Не могу отправить туда вопрос.")

        # Сейчас активен Вариант 1 (отправка напрямую в Telegram чат менеджеров).
        # Если нужен Вариант 2 (Make.com Webhook), закомментируйте Вариант 1
        # и реализуйте Вариант 2, заменив MAKE_WEBHOOK_URL.


# --- Главная функция запуска бота ---
def main() -> None:
    """Запускает бота."""
    # Создаем экземпляр нашего класса QABot
    qa_bot_instance = QABot()

    # Проверяем, загружен ли TELEGRAM_TOKEN перед созданием Updater
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не установлен. Невозможно запустить Updater.")
        return # Прерываем выполнение main если нет токена

    # Создаем Updater и передаем ему токен вашего бота
    # use_context=True требуется для новых версий python-telegram-bot
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Получаем диспетчер для регистрации обработчиков команд и сообщений
    dispatcher = updater.dispatcher

    # Регистрируем обработчики
    # Команда /start
    dispatcher.add_handler(CommandHandler("start", qa_bot_instance.start_command))
    # Команда /help
    dispatcher.add_handler(CommandHandler("help", qa_bot_instance.help_command))
    # Обработка всех текстовых сообщений, кроме команд
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, qa_bot_instance.handle_message))

    # Запускаем бота (начинаем опрос Telegram серверов на наличие новых сообщений)
    logger.info("Бот начал polling...")
    updater.start_polling()

    # Останавливаем бота при получении сигналов SIGINT, SIGTERM or SIGABRT
    updater.idle()

# --- Точка входа при запуске скрипта ---
if __name__ == '__main__':
     main()
