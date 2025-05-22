import chromadb
from chromadb.config import Settings
import openai
import numpy as np
import logging # Импортируем модуль логирования
import os # Для работы с путями
# Импортируем API ключ из config (убедитесь, что config.py находится в корне проекта)
from config import OPENAI_API_KEY

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, db_path="db"):
        logger.info(f"Инициализация Vector Store в директории: {db_path}...")
        # Убедимся, что директория для базы данных существует
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            logger.info(f"Создана директория для базы данных: {db_path}")

        # Инициализация клиента ChromaDB
        # persist_directory указывает, где будут храниться файлы базы данных
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            # Получаем или создаем коллекцию для наших вопросов и ответов
            self.collection = self.client.get_or_create_collection(name="qa_collection")
            logger.info("ChromaDB клиент и коллекция инициализированы.")
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}", exc_info=True)
            self.client = None
            self.collection = None
            return # Прерываем инициализацию при ошибке


        # Настройка API ключа OpenAI
        if not OPENAI_API_KEY:
             logger.warning("OPENAI_API_KEY не установлен. Функции создания эмбеддингов и поиска будут недоступны.")
             self.embedding_model = None
             self.is_openai_ready = False
        else:
            openai.api_key = OPENAI_API_KEY
            # Выбор модели для создания эмбеддингов
            # 'text-embedding-ada-002' - рекомендуемая модель от OpenAI для эмбеддингов
            self.embedding_model = "text-embedding-ada-002"
            self.is_openai_ready = True
            logger.info(f"OpenAI API готов к работе с моделью: {self.embedding_model}")


        logger.info("Vector Store инициализирован.")

    def create_embedding(self, text):
        """
        Создание векторного представления (эмбеддинга) текста с помощью OpenAI API.
        """
        if not self.is_openai_ready:
             # logger.error("OpenAI API не готов. Не могу создать эмбеддинг.") # Это может быть слишком много логов
             return None

        if not text or not isinstance(text, str):
             logger.warning("Попытка создать эмбеддинг для пустого или не строкового текста.")
             return None

        try:
            # OpenAI API принимает список текстов для создания эмбеддингов
            response = openai.Embedding.create(
                input=[text], # Передаем текст в виде списка
                model=self.embedding_model
            )
            # Возвращаем векторное представление первого (и единственного) текста в списке
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга для текста '{text[:50]}...': {e}", exc_info=True)
            return None

    def add_qa_pair(self, question, answer, metadata=None):
        """
        Добавление пары вопрос-ответ в векторную базу.
        """
        if not self.collection or not self.is_openai_ready:
            # logger.error("Vector Store или OpenAI API не готовы. Не могу добавить пару.") # Слишком много логов
            return

        if not question or not answer:
             logger.warning("Попытка добавить пустой вопрос или ответ в базу.")
             return

        # Создаем эмбеддинг для вопроса
        embedding = self.create_embedding(question)
        if embedding is None:
             logger.error(f"Не удалось добавить пару: {question} / {answer[:50]}... из-за ошибки создания эмбеддинга.")
             return

        try:
            # Генерируем уникальный ID для каждого элемента
            # Можно использовать хэш вопроса или просто счетчик
            # Пока используем простой ID на основе количества элементов в коллекции
            # Важно: ID должен быть строкой! Убедимся, что ID уникален
            item_id = f"qa_{self.collection.count() + 1}_{abs(hash(question))}" # Добавляем хэш для уникальности
            # Простая проверка на дубликаты ID (не гарантирует 100% уникальности при параллельных запросах, но для простоты достаточно)
            existing_ids = self.collection.get(ids=[item_id]).get('ids', [])
            if item_id in existing_ids:
                 item_id = f"qa_{self.collection.count() + 1}_{abs(hash(question))}_{np.random.randint(1000)}" # Добавляем случайное число если ID уже есть
                 logger.warning(f"Дубликат ID '{existing_ids[0]}' при добавлении '{question[:50]}...'. Сгенерирован новый ID: '{item_id}'")


            # Добавляем данные в коллекцию ChromaDB
            self.collection.add(
                embeddings=[embedding],      # Список эмбеддингов (один элемент)
                documents=[answer],          # Список документов (ответов)
                metadatas=[metadata] if metadata else [{}], # Список метаданных (если есть)
                ids=[item_id]               # Список уникальных ID
            )
            # logger.info(f"Добавлена пара: '{question[:50]}...'") # Слишком много логов
        except Exception as e:
             logger.error(f"Ошибка при добавлении пары '{question[:50]}...' в ChromaDB: {e}", exc_info=True)


    def search_similar(self, query, n_results=1):
        """
        Поиск наиболее похожих вопросов в базе по запросу.
        Возвращает список найденных документов (ответов) и метаданных.
        """
        if not self.collection or not self.is_openai_ready:
            logger.warning("Vector Store или OpenAI API не готовы. Не могу выполнить поиск.")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

        if not query or not isinstance(query, str):
             logger.warning("Попытка поиска по пустому или не строковому запросу.")
             return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}


        # Создаем эмбеддинг для поискового запроса
        query_embedding = self.create_embedding(query)
        if query_embedding is None:
             logger.error("Не удалось выполнить поиск из-за ошибки создания эмбеддинга запроса.")
             return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

        try:
            # Выполняем поиск в коллекции
            # include=['metadatas', 'documents', 'distances'] - явно указываем, что нужно вернуть
            results = self.collection.query(
                query_embeddings=[query_embedding], # Список эмбеддингов запросов (один элемент)
                n_results=n_results,                # Количество результатов, которые хотим получить
                include=['metadatas', 'documents', 'distances']
            )
            # logger.info(f"Поиск по запросу '{query[:50]}...' завершен. Найдено {len(results.get('documents', []))} результатов.") # Слишком много логов
            # ChromaDB возвращает вложенные списки для каждого запроса.
            # Так как у нас только один запрос, берем первый элемент из каждого списка результатов.
            processed_results = {
                'ids': results.get('ids', [[]])[0],
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'distances': results.get('distances', [[]])[0]
            }

            return processed_results

        except Exception as e:
             logger.error(f"Ошибка при поиске в ChromaDB для запроса '{query[:50]}...': {e}", exc_info=True)
             return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

    def count(self):
        """
        Получение количества элементов в коллекции.
        """
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Ошибка при получении количества элементов в ChromaDB: {e}")
            return 0

    def reset(self):
         """
         Удаление коллекции (сброс базы данных). Используйте осторожно!
         При сбросе удаляются все данные.
         """
         if not self.client:
             logger.error("ChromaDB клиент не инициализирован. Не могу сбросить коллекцию.")
             return
         try:
             self.client.delete_collection(name="qa_collection")
             # После удаления нужно создать коллекцию заново
             self.collection = self.client.get_or_create_collection(name="qa_collection")
             logger.info("Коллекция 'qa_collection' сброшена.")
         except Exception as e:
             logger.error(f"Ошибка при сбросе коллекции ChromaDB: {e}")


# Пример использования (можно удалить после тестирования)
# if __name__ == '__main__':
#     # Для запуска примера убедитесь, что у вас установлен OPENAI_API_KEY в .env
#     # и установлены библиотеки: pip install chromadb openai
#     # Также убедитесь, что директория 'db' может быть создана.
#     logging.basicConfig(level=logging.INFO) # Устанавливаем уровень логирования для примера

#     if OPENAI_API_KEY:
#         logger.info("API ключ OpenAI загружен.")
#         # Сначала сбросим базу для чистоты примера
#         try:
#             chromadb.PersistentClient(path="db").delete_collection(name="qa_collection")
#             logger.info("Старая коллекция удалена для примера.")
#         except:
#              pass # Игнорируем ошибку, если коллекции не было

#         vector_db = VectorStore(db_path="db")

#         if vector_db.is_openai_ready and vector_db.collection:
#             logger.info("\nДобавление тестовых данных:")
#             # Добавляем тестовые данные (можно из Google Sheets загрузить)
#             vector_db.add_qa_pair("Как работает сервис?", "Наш сервис предоставляет услуги ...")
#             vector_db.add_qa_pair("Стоимость подписки?", "Подписка стоит 10$ в месяц.")
#             vector_db.add_qa_pair("Контакты техподдержки?", "Свяжитесь с нами по email: support@example.com")

#             logger.info(f"\nКоличество элементов в базе: {vector_db.count()}")

#             # Выполняем поиск
#             search_query = "Сколько стоит пользование?"
#             search_results = vector_db.search_similar(search_query, n_results=1)

#             if search_results and search_results['documents']:
#                 print(f"\nЗапрос: '{search_query}'")
#                 print(f"Найден ответ: {search_results['documents'][0]}")
#                 print(f"Дистанция: {search_results['distances'][0]}") # Показываем дистанцию
#             else:
#                 print(f"\nПо запросу '{search_query}' ничего не найдено.")

#             # Пример поиска с низким порогом релевантности
#             search_query_irrelevant = "Погода сегодня?"
#             search_results_irrelevant = vector_db.search_similar(search_query_irrelevant, n_results=1)
#             if search_results_irrelevant and search_results_irrelevant['documents']:
#                  print(f"\nЗапрос (нерелевантный): '{search_query_irrelevant}'")
#                  print(f"Найден ответ: {search_results_irrelevant['documents'][0]}")
#                  print(f"Дистанция: {search_results_irrelevant['distances'][0]}") # Показываем дистанцию

#         else:
#              logger.error("Vector Store не готов из-за проблем с OpenAI или ChromaDB.")

#     else:
#          logger.error("OPENAI_API_KEY не установлен в .env. Пример VectorStore не может быть выполнен.")
