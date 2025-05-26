import sqlite3
import os # <--- ДОБАВЛЕН ОБРАТНО
import logging # <--- ДОБАВЛЕН ОБРАТНО
import datetime
import requests

# Импортируем классы и переменные из твоих модулей
from config import TELEGRAM_TOKEN, MANAGER_CHAT_ID, GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID, OPENAI_API_KEY
from utils.google_sheets import GoogleSheetsManager
from database.vector_store import VectorStore

from flask import Flask, request, jsonify
import openai

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Создаем инстанцию Flask приложения ---
app = Flask(__name__)

# --- Функция для чтения инструкций из файла ---
def load_system_instructions(file_path="system_prompt.txt"):
    """Читает системные инструкции из файла."""
    try:
        absolute_file_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Файл инструкций не найден: {absolute_file_path}. Используются инструкции по умолчанию.")
        return "You are a helpful assistant."
    except Exception as e:
        logger.error(f"Ошибка чтения файла инструкций {absolute_file_path}: {e}")
        return "You are a helpful assistant."

# --- Загружаем инструкции при запуске приложения ---
SYSTEM_INSTRUCTIONS_TEXT = load_system_instructions()
logger.info("Системные инструкции загружены.")

# В файле bot.py, можно разместить после импортов и настройки логгера,
# или перед классом QABot.

DB_NAME = "chat_history.db" # Имя файла нашей базы данных

def init_history_db():
    """Инициализирует базу данных и создает таблицу для истории чатов, если она не существует."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info(f"База данных истории '{DB_NAME}' успешно инициализирована.")
    except sqlite3.Error as e:
        logger.error(f"Ошибка при инициализации базы данных истории SQLite: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def add_message_to_history(user_id: str, role: str, content: str):
    """Добавляет сообщение в историю чата для указанного user_id."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (user_id, role, content)
            VALUES (?, ?, ?)
        """, (user_id, role, content))
        conn.commit()
        logger.debug(f"Сообщение от '{role}' для user_id '{user_id}' добавлено в историю.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка при добавлении сообщения в историю SQLite для user_id '{user_id}': {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_recent_history(user_id: str, n_turns: int = 5) -> list:
    """Извлекает последние N пар сообщений (вопрос-ответ) для указанного user_id."""
    history = []
    # Мы хотим получить n_turns * 2 сообщений (вопрос + ответ = 1 ход)
    limit = n_turns * 2 
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        # Сообщения извлекаются в обратном хронологическом порядке (от новых к старым),
        # для OpenAI нам нужен прямой порядок.
        for row in reversed(rows):
            history.append({"role": row[0], "content": row[1]})
        
        logger.debug(f"Извлечено {len(history)} сообщений из истории для user_id '{user_id}'.")
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении истории из SQLite для user_id '{user_id}': {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    return history

# --- Вызов инициализации БД при старте приложения ---
# Это нужно сделать один раз. Можно вызвать init_history_db() сразу после определения,
# или, что лучше, в __init__ класса QABot, если мы хотим связать это с жизненным циклом бота.
# Давай пока вызовем здесь для простоты, чтобы база создалась при запуске скрипта.
init_history_db()

# --- Класс Бота QABot --- (ВОССТАНОВЛЕН)
class QABot:
    def __init__(self):
        if not all([TELEGRAM_TOKEN, MANAGER_CHAT_ID, GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID, OPENAI_API_KEY]):
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: Некоторые необходимые переменные окружения не установлены.")
            # Можно поднять исключение, чтобы остановить запуск, если это критично
            # raise ValueError("Необходимые переменные окружения не установлены для QABot")

        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            logger.info("Ключ OpenAI API успешно установлен для модуля openai.")
        else:
            logger.error("Ключ OPENAI_API_KEY не найден. OpenAI вызовы не будут работать.")

        self.sheets_manager = GoogleSheetsManager(GOOGLE_CREDENTIALS, GOOGLE_SHEETS_ID)
        self.vector_store = VectorStore(db_path="./db") 
        self.load_qa_data()
        logger.info("Экземпляр QABot: sheets_manager и vector_store инициализированы, данные загружены.")

    def load_qa_data(self):
        logger.info("Начало загрузки данных из Google Sheets...")
        qa_data_df = self.sheets_manager.get_qa_data()

        if not qa_data_df.empty:
            logger.info(f"Прочитано {len(qa_data_df)} строк из Google Sheets. Начинаем добавление в векторную базу.")
            self.vector_store.reset() 
            logger.info(f"Старая коллекция в векторной базе очищена перед загрузкой новых данных.")
            
            for index, row in qa_data_df.iterrows():
                question = row.get('Вопрос')
                answer = row.get('Ответ')
                category = row.get('Категория', 'general') 

                if question and answer: 
                    metadata = {'category': category} 
                    self.vector_store.add_qa_pair(question, answer, metadata=metadata)
                else:
                    logger.warning(f"Пропущена строка {index + 2} в Google Sheets из-за отсутствия вопроса или ответа: {row.to_dict()}")
            logger.info(f"Загрузка данных завершена. В векторной базе {self.vector_store.count()} элементов.")
        else:
            logger.warning("Не удалось загрузить данные из Google Sheets или таблица пуста.")
    
    def send_to_manager(self, question: str, user_id: str, user_name: str = "Не указано"):
        """ Отправляет вопрос менеджеру через Webhook Make.com. """
        logger.info(f"Попытка отправки вопроса менеджеру через Make.com от {user_name} (ID: {user_id}): {question}")

        MAKE_WEBHOOK_URL = os.getenv('MAKE_MANAGER_WEBHOOK_URL', "YOUR_MAKE_COM_WEBHOOK_URL_HERE_FOR_MANAGER") 

        if MAKE_WEBHOOK_URL == "YOUR_MAKE_COM_WEBHOOK_URL_HERE_FOR_MANAGER" or not MAKE_WEBHOOK_URL:
            logger.error("URL вебхука Make.com для менеджера не настроен! Не могу отправить вопрос.")
            return False

        payload = {
            "user_id": user_id,
            "user_name": user_name, 
            "question": question,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z" 
        }
        
        try:
            response = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=10) 
            response.raise_for_status() 
            logger.info(f"Вопрос успешно отправлен в Make.com Webhook для менеджера. Статус: {response.status_code}")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Ошибка таймаута при отправке вопроса в Make.com Webhook: {MAKE_WEBHOOK_URL}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при отправке вопроса в Make.com Webhook: {e}", exc_info=True)
            return False


# --- СОЗДАНИЕ ГЛОБАЛЬНОГО ЭКЗЕМПЛЯРА QA-БОТА --- (ВОССТАНОВЛЕН)
logger.info("Попытка создания глобального экземпляра QABot...")
qa_bot_instance = None 
try:
    qa_bot_instance = QABot() 
    logger.info("Глобальный экземпляр QABot успешно создан и готов к работе.")
except ValueError as ve: 
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации QABot (ValueError): {ve}.")
except Exception as e:
    logger.error(f"Непредвиденная КРИТИЧЕСКАЯ ОШИБКА при инициализации QABot: {e}.", exc_info=True)

# --- Определяем маршрут для приема запросов от Make.com ---
# В файле bot.py

# ... (все импорты, включая sqlite3, глобальные переменные, определение QABot, 
#      функции для работы с историей SQLite, создание qa_bot_instance - все это остается как было) ...

@app.route('/webhook', methods=['POST'])
def webhook():
    if not qa_bot_instance:
        logger.error("Экземпляр QABot не был инициализирован. Запрос не может быть обработан.")
        return jsonify({"error": "Внутренняя ошибка сервера: ассистент не инициализирован."}), 500
        
    try:
        data = request.get_json()
        if not data: 
            logger.warning("Получен пустой JSON или не JSON в теле запроса на /webhook.")
            return jsonify({"error": "Request body must be JSON"}), 400

        user_message = data.get('message') 
        # Преобразуем user_id в строку сразу, так как он используется как TEXT в БД истории
        user_id = str(data.get('user_id', 'unknown')) 
        user_name = data.get('user_name', 'Пользователь')

        if not user_message:
            logger.warning("Получен webhook без поля 'message'.")
            return jsonify({"error": "No 'message' field provided in JSON"}), 400

        logger.info(f"Получено сообщение на /webhook от пользователя {user_id} ({user_name}): {user_message}")

        # --- НАЧАЛО ОСНОВНОЙ ЛОГИКИ АССИСТЕНТА ---
        retrieved_context_text = None 
        assistant_reply = None
        
        # 1. Получаем недавнюю историю чата для этого пользователя
        #    n_turns=3 означает, что мы берем 3 последних "хода" (вопрос-ответ), т.е. 6 сообщений
        recent_history = get_recent_history(user_id, n_turns=3) 
        logger.debug(f"Извлеченная история для user_id '{user_id}': {recent_history}")

        try:
            logger.info(f"Ищем контекст для сообщения: '{user_message}'")
            search_results = qa_bot_instance.vector_store.search_similar(user_message, n_results=1)
            distance_threshold = 0.3 

            if search_results and search_results.get('documents') and search_results.get('distances'):
                if search_results['distances'][0] and search_results['documents'][0]:
                    first_distance = search_results['distances'][0][0] 
                    first_document = search_results['documents'][0][0]
                    logger.info(f"Найден ближайший документ с дистанцией: {first_distance:.4f}")
                    if first_distance <= distance_threshold:
                        retrieved_context_text = first_document
                        logger.info(f"Релевантный контекст найден: '{retrieved_context_text}'")
                    else:
                        logger.info(f"Найденный контекст нерелевантен (дистанция {first_distance:.4f} > {distance_threshold}).")
                else:
                    logger.info("Внутренние списки distances/documents в результатах поиска пусты.")
            else:
                logger.info("Результаты поиска из векторной базы пусты или имеют неверный формат.")

            # --- Формируем запрос к OpenAI ---
            # Начинаем с системной инструкции
            messages_for_openai = [{"role": "system", "content": SYSTEM_INSTRUCTIONS_TEXT}]
            
            # Добавляем извлеченную историю чата
            messages_for_openai.extend(recent_history)
            
            # Формируем текущее сообщение пользователя, добавляя контекст, если он есть
            current_user_prompt_content = ""
            if retrieved_context_text:
                current_user_prompt_content = f"Учитывая следующий контекст: \"{retrieved_context_text}\". Ответь на вопрос пользователя: \"{user_message}\""
                logger.info("Контекст будет использован для OpenAI.")
            else:
                current_user_prompt_content = user_message
                logger.info("Контекст не найден или нерелевантен. OpenAI будет вызван без дополнительного контекста из базы знаний (только с историей диалога, если есть).")

            messages_for_openai.append({"role": "user", "content": current_user_prompt_content})
            
            # ВАЖНО: Решение о вызове OpenAI или передаче менеджеру
            # Если даже с учетом истории диалога мы не нашли контекст из БАЗЫ ЗНАНИЙ для текущего вопроса,
            # и если системный промт требует отвечать СТРОГО по базе знаний, то мы можем не вызывать OpenAI.
            # Однако, если мы хотим, чтобы бот мог вести более свободный диалог или отвечать на общие фразы,
            # опираясь на историю, то вызов OpenAI может быть полезен.
            # Твой текущий system_prompt: "Отвечай только на основе информации из базы знаний."
            # и "Если пользователь спрашивает вопрос, которого нет в базе знаний, отвечай: 'Извините, я не владею такой информацией...'"
            # Это означает, что если retrieved_context_text НЕТ, то мы НЕ должны вызывать OpenAI для генерации ответа по теме.
            
            if retrieved_context_text: # Вызываем OpenAI только если есть релевантный контекст из БАЗЫ ЗНАНИЙ
                try:
                    logger.info(f"Отправка запроса в OpenAI с моделью gpt-3.5-turbo. Сообщений в истории: {len(recent_history)}")
                    openai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=messages_for_openai,
                        temperature=0.7 
                    )
                    assistant_reply = openai_response.choices[0].message['content'].strip()
                    logger.info(f"Ответ от OpenAI получен: {assistant_reply}")
                except Exception as openai_error:
                    logger.error(f"Ошибка при вызове OpenAI API: {openai_error}", exc_info=True)
                    assistant_reply = "Извините, произошла ошибка при обращении к AI-ассистенту. Попробуйте позже."
            else: 
                logger.info("Релевантный контекст из базы знаний не найден. Формируем ответ о передаче менеджеру.")
                assistant_reply = "Извините, я не владею такой информацией по вашему вопросу. Ваш вопрос будет передан менеджеру, и он обязательно вам ответит."
                # Вызываем отправку менеджеру
                logger.info(f"Передаем вопрос менеджеру: '{user_message}' от пользователя {user_id} ({user_name})")
                send_status = qa_bot_instance.send_to_manager(
                    question=user_message, 
                    user_id=user_id, # user_id уже строка 
                    user_name=user_name
                )
                if send_status:
                    logger.info("Вопрос успешно поставлен в очередь на отправку менеджеру через Make.com.")
                else:
                    logger.error("Не удалось поставить вопрос в очередь на отправку менеджеру через Make.com.")

        except Exception as assistant_logic_error:
            logger.error(f"Ошибка в основной логике ассистента: {assistant_logic_error}", exc_info=True)
            assistant_reply = "Извините, произошла внутренняя ошибка при обработке вашего запроса."
            # В этом случае, сохраняем вопрос пользователя, но ответ об ошибке
            add_message_to_history(user_id, "user", user_message)
            add_message_to_history(user_id, "assistant", assistant_reply)
            return jsonify({"reply": assistant_reply, "error_details": str(assistant_logic_error)}), 500
        
        # --- Сохраняем текущий диалог в историю ---
        # Сохраняем сообщение пользователя
        add_message_to_history(user_id, "user", user_message)
        # Сохраняем ответ ассистента, если он был сгенерирован
        if assistant_reply:
            add_message_to_history(user_id, "assistant", assistant_reply)
        else: # На случай если assistant_reply по какой-то причине None
            logger.error("assistant_reply is None перед финальным return. Это не должно было произойти.")
            assistant_reply = "Не удалось обработать ваш запрос. Пожалуйста, попробуйте еще раз."
            add_message_to_history(user_id, "assistant", assistant_reply) # Сохраняем и этот ответ

        return jsonify({"reply": assistant_reply})

    except Exception as e: 
        logger.error("Общая ошибка при обработке /webhook запроса:", exc_info=True)
        # На этом уровне ошибке не логируем user_message в историю, т.к. ошибка могла быть до его обработки
        return jsonify({"error": "Internal server error"}), 500

# ... (остальной код файла bot.py ниже остается без изменений)
