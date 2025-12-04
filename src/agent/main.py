import asyncio
import re
from pprint import pprint
from typing import Dict, Optional
from datetime import datetime
import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

# Загружаем переменные окружения
load_dotenv()


# Конфигурация
class Settings(BaseSettings):
    bot_token: str = Field(..., env="BOT_TOKEN")
    opennotebook_url: str = Field(default="http://localhost:5055", env="OPENNOTEBOOK_URL")
    notebook_id: str = Field(..., env="NOTEBOOK_ID")

    class Config:
        env_file = ".env"


# Модели данных
class ChatMessage(BaseModel):
    user_id: int
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionManager:
    """Управление единой сессией для всех пользователей"""

    def __init__(self, opennotebook_url: str, notebook_id: str):
        self._session_id: Optional[str] = None
        self._context: Optional[dict] = None
        self.opennotebook_url = opennotebook_url
        self.notebook_id = notebook_id
        self._lock = asyncio.Lock()

    async def get_or_create_session(self) -> tuple[str, dict]:
        """Получить или создать единую сессию и контекст для всех пользователей"""
        async with self._lock:
            if self._session_id and self._context:
                return self._session_id, self._context

            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Создаем сессию
                session_response = await client.post(
                    f"{self.opennotebook_url}/api/chat/sessions",
                    json={
                        "notebook_id": self.notebook_id,
                        "title": "Стратсессия"
                    }
                )
                pprint(session_response)
                session_response.raise_for_status()
                session_data = session_response.json()
                self._session_id = session_data["id"]

                logger.info(f"Создана общая сессия {self._session_id}")

                # 2. Получаем контекст
                context_response = await client.post(
                    f"{self.opennotebook_url}/api/chat/context",
                    json={
                        "notebook_id": self.notebook_id,
                        "context_config": {}
                    }
                )
                context_response.raise_for_status()
                self._context = context_response.json()

                logger.info("Получен контекст для общей сессии")

                return self._session_id, self._context

    def clear_session(self):
        """Очистить сессию"""
        self._session_id = None
        self._context = None

def format_markdown_for_telegram(text: str) -> str:
    lines = text.splitlines()
    output_lines = []
    in_table = False
    table_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Заголовки
        if line.startswith("### "):
            content = line[4:].strip()
            output_lines.append(f"\n🔹 {content}")
            i += 1
            continue
        elif line.startswith("## "):
            content = line[3:].strip()
            output_lines.append(f"\n📌 {content}")
            i += 1
            continue
        elif line.startswith("# "):
            content = line[2:].strip()
            output_lines.append(f"\n🎯 {content.upper()}")
            i += 1
            continue

        # Начало таблицы
        if "|" in line and not in_table:
            if i + 1 < len(lines) and re.search(r'\|.*?-.*?\|', lines[i + 1]):
                in_table = True
                table_lines = [line]
                i += 1
                continue

        if in_table:
            table_lines.append(line)
            if "|" not in line or i == len(lines) - 1:
                output_lines.extend(_convert_table_to_plain_bullets(table_lines))
                in_table = False
                table_lines = []
            i += 1
            continue

        # Убираем ** и __
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        line = re.sub(r'__(.*?)__', r'\1', line)

        # Убираем ссылки [текст](url) → текст
        line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)

        output_lines.append(line)
        i += 1

    result = "\n".join(output_lines).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def _convert_table_to_plain_bullets(table_lines: list[str]) -> list[str]:
    if len(table_lines) < 2:
        return ["\n" + "\n".join(table_lines)]

    headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
    data_lines = table_lines[2:]

    if not headers:
        return ["\n" + "\n".join(table_lines)]

    result = ["\n"]
    for row in data_lines:
        if not row.strip() or '---' in row:
            continue
        cells = [c.strip() for c in row.split('|')[1:-1]]
        if len(cells) != len(headers):
            continue

        # Формат: • Заголовок: остальные колонки через запятую
        main = cells[0]
        rest = " | ".join(cells[1:])
        result.append(f"• {main}: {rest}")

    return result


class MessageQueue:
    """Очередь сообщений для синхронной обработки"""

    def __init__(self, opennotebook_url: str, notebook_id: str):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.opennotebook_url = opennotebook_url
        self.notebook_id = notebook_id
        self.session_manager = SessionManager(opennotebook_url, notebook_id)
        self.lock = asyncio.Lock()

    async def add_message(self, bot: Bot, chat_id: int, user_id: int, message: str):
        """Добавить сообщение в очередь"""
        # Проверяем, обрабатывается ли сейчас какое-то сообщение
        async with self.lock:
            is_processing = self.processing
            queue_size = self.queue.qsize()

        if is_processing or queue_size > 0:
            position = queue_size + 1
            await bot.send_message(
                chat_id,
                f"⏳ Пожалуйста, подождите, ваша инициатива в очереди на обработку.\n"
                f"Позиция в очереди: {position}"
            )

        chat_msg = ChatMessage(user_id=user_id, message=message)
        await self.queue.put((bot, chat_id, user_id, chat_msg))

        # Запускаем обработку, если она еще не запущена
        async with self.lock:
            if not self.processing:
                self.processing = True
                asyncio.create_task(self.process_queue())

    async def process_queue(self):
        """Обработка очереди сообщений"""
        logger.info("Начало обработки очереди сообщений")

        try:
            while not self.queue.empty():
                bot, chat_id, user_id, chat_msg = await self.queue.get()

                try:
                    await bot.send_chat_action(chat_id, "typing")
                    response = await self.send_to_opennotebook(user_id, chat_msg.message)

                    # Форматируем для Telegram
                    formatted_response = format_markdown_for_telegram(response)

                    # Отправляем ответ пользователю
                    await self.send_long_message(bot, chat_id, formatted_response)

                except Exception as e:
                    logger.error(f"Ошибка обработки сообщения от пользователя {user_id}: {e}")
                    await bot.send_message(
                        chat_id,
                        f"❌ Произошла ошибка при обработке вашей инициативы: {str(e)}"
                    )

                finally:
                    self.queue.task_done()

        finally:
            async with self.lock:
                self.processing = False
            logger.info("Обработка очереди завершена")

    async def send_long_message(self, bot: Bot, chat_id: int, text: str):
        """Отправка длинного сообщения в виде нескольких частей, если превышен лимит длины сообщения в Telegram"""
        TELEGRAM_MAX_LENGTH = 4096
        
        if len(text) <= TELEGRAM_MAX_LENGTH:
            await bot.send_message(chat_id, text)
            return
        
        # Разбиваем текст на части, не превышающие лимит
        parts = []
        current_part = ""
        
        # Разбиваем по строкам, чтобы не резать посреди строк
        lines = text.split('\n')
        
        for line in lines:
            if len(current_part + line + '\n') <= TELEGRAM_MAX_LENGTH:
                current_part += line + '\n'
            else:
                if current_part:
                    parts.append(current_part.rstrip('\n'))
                current_part = line + '\n'
        
        if current_part:
            parts.append(current_part.rstrip('\n'))
        
        # Отправляем все части
        for i, part in enumerate(parts):
            if i == 0:
                # Для первого сообщения не добавляем префикс
                await bot.send_message(chat_id, part)
            else:
                # Для последующих частей добавляем номер части
                await bot.send_message(chat_id, f"({i + 1}/{len(parts)})\n{part}")
            
            # Небольшая задержка, чтобы не превысить рейт-лимит
            await asyncio.sleep(0.1)

    async def send_to_opennotebook(self, user_id: int, message: str) -> str:
        """Отправка сообщения в Open-Notebook"""
        # Получаем или создаем единую сессию для всех пользователей
        session_id, context = await self.session_manager.get_or_create_session()

        payload = {
            "session_id": session_id,
            "message": message,
            "context": context
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                logger.info(f"Отправка запроса в Open-Notebook от пользователя {user_id}")
                logger.debug(f"Payload: {payload}")

                response = await client.post(
                    f"{self.opennotebook_url}/api/chat/execute",
                    json=payload
                )
                response.raise_for_status()

                data = response.json()

                # Извлекаем последнее сообщение от AI
                messages = data.get("messages", [])

                if messages:
                    return messages[-1].get("content", "Ответ не получен")
                else:
                    return "Ответ от AI не получен"

            except httpx.HTTPError as e:
                logger.error(f"HTTP ошибка: {e}")
                raise Exception(f"Ошибка связи с Open-Notebook: {str(e)}")
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                raise


# Инициализация бота
settings = Settings()
bot = Bot(token=settings.bot_token)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
message_queue = MessageQueue(settings.opennotebook_url, settings.notebook_id)


# Обработчики команд
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    await message.answer(
        "👋 Добро пожаловать в бот для стратегических инициатив!\n\n"
        "Отправьте мне вашу инициативу, и я обработаю её через Open-Notebook.\n"
        "Если поступит несколько запросов одновременно, они будут обработаны последовательно.\n\n"
        "Команды:\n"
        "/start - Начать работу\n"
        "/help - Показать справку"
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Справка"""
    await message.answer(
        "ℹ️ Справка по использованию бота:\n\n"
        "Просто отправьте мне текст вашей стратегической инициативы, "
        "и я передам её на анализ в Open-Notebook.\n\n"
        "Все сообщения обрабатываются последовательно, "
        "поэтому если кто-то уже отправил запрос, "
        "вам придется немного подождать в очереди."
    )


@dp.message(F.text)
async def handle_message(message: types.Message):
    """Обработка текстовых сообщений"""
    if not message.text:
        return

    logger.info(f"Получено сообщение от пользователя {message.from_user.id}: {message.text[:50]}...")

    await message_queue.add_message(
        bot=bot,
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        message=message.text
    )


async def main():
    """Запуск бота"""
    logger.info("Запуск бота...")
    logger.info(f"Open-Notebook URL: {settings.opennotebook_url}")
    logger.info(f"Notebook ID: {settings.notebook_id}")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())