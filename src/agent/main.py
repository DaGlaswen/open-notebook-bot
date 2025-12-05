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
from aiogram.enums import ParseMode

# Загружаем переменные окружения
load_dotenv()


# Конфигурация
class Settings(BaseSettings):
    bot_token: str = Field(..., env="BOT_TOKEN")
    opennotebook_url: str = Field(default="http://localhost:5055", env="OPENNOTEBOOK_URL")
    notebook_id: str = Field(..., env="NOTEBOOK_ID")
    session_id: str = Field(default="chat_session:30nsl9iu3k6pvnovcn8w", env="SESSION_ID")

    class Config:
        env_file = ".env"


# Модели данных
class ChatMessage(BaseModel):
    user_id: int
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionManager:
    """Управление единой сессией для всех пользователей"""

    def __init__(self, opennotebook_url: str, notebook_id: str, session_id: str):
        self._session_id: Optional[str] = session_id  # Используем захардкоженный session_id
        self._context: Optional[dict] = None
        self.opennotebook_url = opennotebook_url
        self.notebook_id = notebook_id
        self._lock = asyncio.Lock()

    async def get_or_create_session_and_context(self) -> tuple[str, dict]:
        """Получить или создать единую сессию и контекст для всех пользователей"""
        async with self._lock:
            if self._context:
                return self._session_id, self._context

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Получаем контекст для существующей сессии
                context_response = await client.post(
                    f"{self.opennotebook_url}/api/chat/context",
                    json={
                        "notebook_id": self.notebook_id,
                        "context_config": {}
                    }
                )
                context_response.raise_for_status()
                self._context = context_response.json()

                logger.info(f"Получен контекст для сессии {self._session_id}")

                return self._session_id, self._context

    def clear_session(self):
        """Очистить контекст (сессия остается захардкоженной)"""
        self._context = None


def format_markdown_for_telegram_html(text: str) -> str:
    """
    Преобразует Markdown в HTML, безопасный для Telegram.
    """
    lines = text.splitlines()
    output_lines = []
    in_table = False
    table_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Экранируем HTML-символы заранее, чтобы не сломать разметку
        safe_line = html.escape(line)

        # Заголовки
        if line.startswith("### "):
            content = html.escape(line[4:].strip())
            output_lines.append(f"\n<b>{content}</b>")
            i += 1
            continue
        elif line.startswith("## "):
            content = html.escape(line[3:].strip())
            output_lines.append(f"\n<b>📌 {content}</b>")
            i += 1
            continue
        elif line.startswith("# "):
            content = html.escape(line[2:].strip())
            output_lines.append(f"\n<b>🎯 {content.upper()}</b>")
            i += 1
            continue

        # Обработка таблиц (включая структурированные данные с |)
        if "|" in line and not in_table:
            # Проверяем, является ли это таблицей по наличию разделителя
            if i + 1 < len(lines) and re.search(r'\|.*?-.*?\|', lines[i + 1]):
                in_table = True
                table_lines = [line]
                i += 1
                continue
            # Также проверяем, есть ли похожие строки после (для нестандартных таблиц)
            elif i + 1 < len(lines) and "|" in lines[i + 1]:
                in_table = True
                table_lines = [line]
                i += 1
                continue

        if in_table:
            table_lines.append(line)
            # Прекращаем сбор таблицы если:
            # 1. Следующая строка не содержит | и не пустая
            # 2. Это последняя строка
            # 3. Следующая строка выглядит как заголовок или другая структура
            if (i == len(lines) - 1 or
                    ("|" not in lines[i + 1] and lines[i + 1].strip()) or
                    lines[i + 1].startswith(("###", "##", "#", "• ", "- ", "* "))):
                output_lines.extend(_convert_table_to_html_bullets(table_lines))
                in_table = False
                table_lines = []
                i += 1  # Пропускаем эту строку, т.к. она не принадлежит таблице
                continue
            else:
                i += 1
            continue

        # Жирный текст: **текст** → <b>текст</b>
        safe_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_line)
        safe_line = re.sub(r'__(.*?)__', r'<b>\1</b>', safe_line)

        # Ссылки [текст](url) → текст
        safe_line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', safe_line)

        # Обработка списков с маркерами •
        if line.strip().startswith("• "):
            safe_line = re.sub(r'^\s*•\s*', '• ', safe_line)
            output_lines.append(safe_line)
        else:
            output_lines.append(safe_line)
        i += 1

    result = "\n".join(output_lines).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


import re
import html


def _convert_table_to_html_bullets(table_lines: list[str]) -> list[str]:
    if len(table_lines) < 1:
        # Просто экранируем и возвращаем как есть
        escaped = [html.escape(line) for line in table_lines]
        return ["\n" + "\n".join(escaped)]

    try:
        # Проверяем, действительно ли это таблица (с разделителями |)
        has_table_format = any("|" in line for line in table_lines)

        if has_table_format:
            # Обработка стандартной таблицы
            headers = []
            data_lines = []

            for idx, line in enumerate(table_lines):
                if idx == 0 and "|" in line:
                    headers = [h.strip() for h in line.split('|')[1:-1]]
                elif idx == 1 and re.search(r'\|.*?-.*?\|', line):
                    # Это строка с разделителями ---, пропускаем
                    continue
                elif "|" in line:
                    data_lines.append(line)

            if headers and data_lines:
                def markdown_bold_to_html(text: str) -> str:
                    """Заменяет **текст** и __текст__ на <b>текст</b>"""
                    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
                    return text

                def escape_preserving_b_tags(text: str) -> str:
                    """Экранирует HTML, но оставляет <b> и </b> нетронутыми"""
                    parts = re.split(r'(<b>.*?</b>)', text)
                    for i, part in enumerate(parts):
                        if not (part.startswith('<b>') and part.endswith('</b>')):
                            parts[i] = html.escape(part)
                    return ''.join(parts)

                result = ["\n"]
                for row in data_lines:
                    if not row.strip() or '---' in row:
                        continue
                    cells = [c.strip() for c in row.split('|')[1:-1]]
                    if len(cells) != len(headers):
                        continue

                    # Обрабатываем первую ячейку (жирный заголовок пункта)
                    main_raw = cells[0]
                    main_with_bold = markdown_bold_to_html(main_raw)
                    main_safe = escape_preserving_b_tags(main_with_bold)

                    # Обрабатываем остальные ячейки (могут тоже содержать **...**)
                    rest_parts = []
                    for c in cells[1:]:
                        c_with_bold = markdown_bold_to_html(c)
                        c_safe = escape_preserving_b_tags(c_with_bold)
                        rest_parts.append(c_safe)
                    rest = " | ".join(rest_parts)

                    result.append(f"• {main_safe}: {rest}")
                return result
            else:
                # Это не стандартная таблица, обрабатываем как список с разделителями
                result = ["\n"]
                for line in table_lines:
                    if "|" in line:
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        if parts:
                            result.append(f"• {' | '.join(parts)}")
                    else:
                        result.append(html.escape(line))
                return result
        else:
            # Это не таблица, просто список строк
            result = ["\n"]
            for line in table_lines:
                result.append(html.escape(line))
            return result

    except Exception:
        # Fallback: экранировать всё как plain text
        escaped = [html.escape(line) for line in table_lines]
        return ["\n" + "\n".join(escaped)]


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

    def __init__(self, opennotebook_url: str, notebook_id: str, session_id: str):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.opennotebook_url = opennotebook_url
        self.notebook_id = notebook_id
        self.session_manager = SessionManager(opennotebook_url, notebook_id, session_id)
        self.lock = asyncio.Lock()

    async def add_message(self, bot: Bot, chat_id: int, user_id: int, message: str, is_list_command: bool = False):
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
        await self.queue.put((bot, chat_id, user_id, chat_msg, is_list_command))

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
                bot, chat_id, user_id, chat_msg, is_list_command = await self.queue.get()

                try:
                    await bot.send_chat_action(chat_id, "typing")

                    # Формируем сообщение с информацией о пользователе
                    if is_list_command:
                        # Для команды /list используем специальный промпт
                        message_with_user = chat_msg.message
                    else:
                        # Для обычных сообщений добавляем информацию о пользователе
                        message_with_user = f"Пользователь {user_id}: {chat_msg.message}"

                    response = await self.send_to_opennotebook(user_id, message_with_user)

                    logger.debug(f"Ответ от Open-notebook для user_id {user_id}: {response}")

                    # Форматируем для Telegram
                    formatted_response = format_markdown_for_telegram_html(response)

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

    from aiogram.enums import ParseMode

    async def send_long_message(self, bot: Bot, chat_id: int, text: str):
        """Отправка длинного HTML-сообщения"""
        TELEGRAM_MAX_LENGTH = 4096

        if len(text) <= TELEGRAM_MAX_LENGTH:
            await bot.send_message(chat_id, text, parse_mode=ParseMode.HTML)
            return

        # Разбиваем, но НЕ по символам, а по строкам, чтобы не разорвать теги
        parts = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) <= TELEGRAM_MAX_LENGTH:
                current += line
            else:
                if current:
                    parts.append(current)
                current = line
        if current:
            parts.append(current)

        for i, part in enumerate(parts):
            try:
                if i == 0:
                    await bot.send_message(chat_id, part, parse_mode=ParseMode.HTML)
                else:
                    await bot.send_message(chat_id, f"({i + 1}/{len(parts)})\n{part}", parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning(f"HTML parse error, fallback to plain text: {e}")
                # Fallback: убираем HTML-теги
                plain = re.sub(r'<[^>]+>', '', part)
                if i == 0:
                    await bot.send_message(chat_id, plain)
                else:
                    await bot.send_message(chat_id, f"({i + 1}/{len(parts)})\n{plain}")
            await asyncio.sleep(0.1)

    async def send_to_opennotebook(self, user_id: int, message: str) -> str:
        """Отправка сообщения в Open-Notebook"""
        # Получаем или создаем единую сессию для всех пользователей
        session_id, context = await self.session_manager.get_or_create_session_and_context()

        payload = {
            "session_id": session_id,
            "message": message,
            "context": context
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                logger.info(f"Отправка запроса в Open-Notebook от пользователя {user_id}: {message}")

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
message_queue = MessageQueue(settings.opennotebook_url, settings.notebook_id, settings.session_id)


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
        "/list - Показать текущую иерархию инициатив\n"
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
        "вам придется немного подождать в очереди.\n\n"
        "Команды:\n"
        "/list - Показать текущую иерархию инициатив"
    )


@dp.message(Command("list"))
async def cmd_list(message: types.Message):
    """Обработчик команды /list"""
    user_id = message.from_user.id
    prompt = "Дай текущую иерархию списка инициатив из чата (порядковый номер инициативы/название/RICE score/пользователь), где пользователь - это ID пользователя, который передавался в предыдущих сообщениях. Бери только те инициативы, которые были предложены в чате, не используй информацию из приложенных документов."

    logger.info(f"Получена команда /list от пользователя {user_id}")

    await message_queue.add_message(
        bot=bot,
        chat_id=message.chat.id,
        user_id=user_id,
        message=prompt,
        is_list_command=True
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
        message=message.text,
        is_list_command=False
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