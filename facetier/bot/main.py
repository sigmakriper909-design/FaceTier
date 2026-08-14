import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo, MenuButtonWebApp
from aiogram.enums import ParseMode
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN", "8824474369:AAEHqmoA6JekZJ-89206m2iDr_-rG7Vll5w")

# Когда задеплоим Mini App — сюда поставим реальный HTTPS URL
MINIAPP_URL = os.getenv("MINIAPP_URL", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "<b>FaceTier</b>\n\n"
        "Жёсткий и честный разбор твоего лица.\n"
        "Загрузи анфас + профиль — получи оценки по зонам, потенциал и приоритеты прокачки.\n\n"
        "<i>Сейчас бот в тестовом режиме (всё бесплатно).</i>"
    )

    if MINIAPP_URL:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🚀 Открыть разбор",
                web_app=WebAppInfo(url=MINIAPP_URL)
            )]
        ])
    else:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Начать разбор (скоро)", callback_data="soon")]
        ])

    await message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data == "soon")
async def soon_callback(callback: types.CallbackQuery):
    await callback.answer("Mini App почти готов, подожди немного", show_alert=True)


@dp.message(Command("status"))
async def cmd_status(message: Message):
    await message.answer(
        f"Бот работает ✅\n"
        f"Mini App URL: {MINIAPP_URL or 'ещё не установлен'}"
    )


async def main():
    logger.info("Starting FaceTier bot (@Facerst_bot)...")
    if MINIAPP_URL:
        try:
            await bot.set_chat_menu_button(
                menu_button=MenuButtonWebApp(
                    text="Разбор",
                    web_app=WebAppInfo(url=MINIAPP_URL)
                )
            )
        except Exception as e:
            logger.warning(f"Не удалось поставить menu button: {e}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
