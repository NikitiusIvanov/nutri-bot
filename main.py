import io
import asyncio
import sys
import os
import datetime
from typing import Any, Awaitable, Callable, Dict
import numpy as np
import logging
import matplotlib.pyplot as plt

# Google Generative AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Image
from aiohttp import web

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql import text

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart
from aiogram import F
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    TelegramObject,
    KeyboardButton,
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    BufferedInputFile
)
from aiogram.utils.keyboard import (
    ReplyKeyboardBuilder,
    InlineKeyboardBuilder
)
from aiogram.client.session.aiohttp import AiohttpSession
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# get db credentials
# Db Configuration
POSTGRES_USER=os.getenv('POSTGRES_USER')
POSTGRES_DB=os.getenv('POSTGRES_DB')
POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST=os.getenv('POSTGRES_HOST')
POSTGRES_PORT='5432'

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "database": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD
}

DB_URL = (
    'postgresql+asyncpg://'
    f'{POSTGRES_USER}:{POSTGRES_PASSWORD}'
    f'@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
)

# Create async engine and session pool
engine = create_async_engine(
    DB_URL, 
    echo='debug', 
    echo_pool='debug', 
    pool_size=20
)

session_maker = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# get the credentials from env vars
BOT_TOKEN = os.getenv('BOT_TOKEN')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')

BASE_WEBHOOK_URL = os.getenv('BASE_WEBHOOK_URL')
# Webserver settings
WEB_SERVER_HOST = "0.0.0.0"
# Port for incoming request
WEB_SERVER_PORT = 8080
# Path to webhook route, on which Telegram will send requests
WEBHOOK_PATH = "/webhook"
      
####################### google AI API settings #######################    
# Settings for vertexai initialization
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION')

if PROJECT_ID is not None:
    vertexai.init(project=PROJECT_ID, location=REGION)

    generation_config = {
        'temperature': 0,
    }

    model = GenerativeModel("gemini-1.5-pro-002", generation_config=generation_config)


####################### Set prompt for tasks #######################
PROMPT = """
You are a helpful AI assistant that helps people collect data about their diets 
by food photos that their sending to you.
Recognize food in this picture and
give your estimation of the:
 * total amount of calories in kkal, 
 * mass in grams, 
 * fat in grams,
 * carbonhydrates in grams

If you reconize there is no food in the photo 
write your answer by following format and nothing more:

no food

If you reconize some food on the photo use low-high borders 
for you estimation of the nutritional facts
and write your answer by following format and nothing more:

dish_name: apple pie
calories: 230, 240
mass: 340, 350
fat: 5.0, 5.5
carb: 22, 25
protein: 24, 25
"""

####################### Data processing #######################
DATA_PROCESSING_CHAPTER = None
async def sql_check_if_user_exists(
    session: AsyncSession, 
    user_id: int,
):
    """Checks if a user with the given user_id exists in the database.

    Args:
        con: An asyncpg connection.
        user_id: The user ID to check.

    Returns:
        True if the user exists, False otherwise.
    """

    result = await session.execute(
        text("SELECT EXISTS (SELECT 1 FROM users WHERE user_id = :user_id)"), 
        {'user_id': user_id}
    )
    result = result.fetchone()

    return result[0]


async def sql_get_latest_daily_calories_goal(
    session: AsyncSession, 
    user_id: int,
):
    result = await session.execute(
        text("""
        SELECT daily_calories_goal
        FROM users
        WHERE user_id = :user_id
        ORDER BY timestamp DESC
        LIMIT 1
        """), 
        {'user_id': user_id}
    )

    return result.fetchall()


async def sql_write_new_user(
    session: AsyncSession, 
    user_row: dict
):
    """
    """
    await session.execute(
        text("""
        INSERT INTO users (
            first_name,
            last_name,
            user_name,
            user_id,
            chat_id,
            height,
            weight,
            age,
            daily_calories_goal
        ) VALUES (
            :first_name,
            :last_name,
            :user_name,
            :user_id,
            :chat_id,
            :height,
            :weight,
            :age,
            :daily_calories_goal
       )
        """),
        {
            'first_name': user_row['first_name'],
            'last_name': user_row['last_name'],
            'user_name': user_row['user_name'],
            'user_id': user_row['user_id'],
            'chat_id': user_row['chat_id'],
            'height': user_row['height'],
            'weight': user_row['weight'],
            'age': user_row['age'],
            'daily_calories_goal': user_row['daily_calories_goal']
        }
    )

    await session.commit()


async def sql_write_nutrition(
    session: AsyncSession, 
    meal_row: dict
):
    """
    """
    await session.execute(
        text("""
        INSERT INTO meals (
            user_id,
            dish_name,
            calories,
            mass,
            protein,
            carb,
            fat 
        ) VALUES (
            :user_id,
            :dish_name,
            :calories,
            :mass,
            :protein,
            :carb,
            :fat
        )
        """),
        {
            'user_id': meal_row['user_id'],
            'dish_name': meal_row['dish_name'],
            'calories': meal_row['calories'],
            'mass': meal_row['mass'],
            'protein': meal_row['protein'],
            'carb': meal_row['carb'],
            'fat': meal_row['fat']
        }
    )

    await session.commit()


async def sql_check_daily_goal_exists(
    session: AsyncSession, 
    user_id: int,
) -> bool:
    query = text("""
    SELECT EXISTS (
        SELECT 1 FROM users 
        WHERE user_id = :user_id
        and daily_calories_goal is not null
    )
    """)
    result = await session.execute(query, {'user_id': user_id})
    result = result.fetchone()

    return result[0]


async def sql_get_daily_goal(
    session: AsyncSession,
    user_id: int,
):
    query_daily_goal = text(
        """
        SELECT daily_calories_goal
        FROM users
        WHERE user_id = :user_id
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )

    result = await session.execute(
        query_daily_goal,
        {'user_id': user_id}
    )
    daily_calories_goal_result = result.scalar()

    print('daily goal query result', daily_calories_goal_result)
    
    return daily_calories_goal_result


async def sql_get_user_todays_statistics(
    session: AsyncSession,
    user_id: int,
):

    query_todays_nutrition = text(
        """
        SELECT 
            SUM(calories),
            SUM(protein),
            SUM(carb),
            SUM(fat)
        FROM meals
        WHERE 
            timestamp::date = CURRENT_DATE
            AND 
            user_id = :user_id
        GROUP BY user_id;
        """
    )

    todays_nutrition_result = await session.execute(
        query_todays_nutrition,
        {'user_id': user_id}
    )
    
    return todays_nutrition_result.fetchall()


async def check_user_exist(
    message: Message,
    session: AsyncSession,
):

    is_user_exist = await sql_check_if_user_exists(session, message.from_user.id)

    if is_user_exist == False:
        
        first_name = message.from_user.first_name

        last_name = message.from_user.last_name

        user_name = message.from_user.username

        user_id = message.from_user.id

        timestamp = datetime.datetime.now().astimezone().isoformat()

        chat_id = message.chat.id

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_new_user(
            session=session, 
            user_row={
                'first_name': first_name,
                'last_name': last_name,
                'user_name': user_name,
                'user_id': user_id,
                'timestamp': timestamp,
                'chat_id': chat_id,
                'height': height,
                'weight': weight,
                'age': age,
                'daily_calories_goal': daily_calories_goal,
            }
        )


def response_to_dict(
    response: str,
) -> dict[str, list[float]] | str:
    """
    transform string 
    "dish_name: apple pie
    calories: 15, 25
    mass: 100, 150
    fat: 0.3, 0.5
    carb: 2, 4
    protein: 2, 3"
    -> to dict = {
        calories: 20,
        mass: 125,
        fat: 0.4,
        carb: 3,
        protein: 2.5,
        name: apple pie,
    }
    """
    if response.text == 'no food':
        return response.text
    else:
        try:
            result = {
                row.split(': ')[0]: np.mean(
                    [
                        round(float(row.split(': ')[1].split(',')[0]), 2),
                        round(float(row.split(': ')[1].split(',')[1]), 2)
                    ]
                )
                for row in response.text.strip('\n').split('\n')[1:]
            }
            result['dish_name'] = (
                response.text
                .strip('\n')
                .split('\n')[0]
                .split(': ')[1]
            )
            assert set(
                [
                    'dish_name', 'calories', 'mass', 'fat', 'carb', 'protein'
                ]
            ) == set(result.keys())
        except:
            result = 'not correct result'
    return result


def today_statistic_plotter(
    daily_calories_goal,
    total_calories,
    total_protein,
    total_carb,
    total_fat
) -> tuple[io.BytesIO, plt.figure]:
    plt.style.use('fast')
    # Create a figure and set size
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # First plot: Calories
    categories = ['Calories']
    daily_goals = [daily_calories_goal]
    consumed = [total_calories]

    # Plot daily goal and consumed calories
    width = 0.3
    axs[0].bar(categories, daily_goals, width=width, label='Daily goal', color='gray', alpha=0.6)
    axs[0].bar(categories, consumed, width=width, label='Today\'s calories', color='green')

    axs[0].set_title('Calories (kcal)')
    axs[0].legend()

    # Second plot: Macronutrients (protein, carb, fat)
    nutrients = ['Protein', 'Carb', 'Fat']
    values = [total_protein, total_carb, total_fat]
    colors = ['brown', 'purple', 'orange']

    axs[1].bar(nutrients, values, color=colors)
    axs[1].set_title('Macronutrients (g)')

    # Adjust layout
    plt.tight_layout()

    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind buffer to the beginning for reading

    return buf, fig


def message_lenght(message_text: str | None):
    """Returns lenght of the recieved message
    """
    if message_text is not None:
        return len(message_text)
    else:
        return None

####################### Bot logic #######################
BOT_LOGIC_CHAPTER = None

class Form(StatesGroup):
    chat_id = State()
    photo_ok = State()
    nutrition_ok = State()
    nutrition_facts = State()
    username = State()
    first_name = State()
    last_name = State()
    user_id = State()
    edit_request = State()
    key_to_edit = State()
    new_value = State()
    edit_daily_goal = State()
    statistics = State()

form_router = Router()

def text_from_nutrition_facts(
    nutrition_facts: dict[str, str|float],
    is_saved: bool=False,
) -> str:
    text = (
        '*Here is my estimation of the nutrition facts about your photo:*\n'
        f'🍽 Dish name: *{nutrition_facts["dish_name"]}*\n'
        f'🧮 Total calories: *{round(nutrition_facts["calories"], 2)}* kcal\n'
        f'⚖️ Total mass: *{round(nutrition_facts["mass"], 2)}* g\n'
        f'🍖 Proteins mass: *{round(nutrition_facts["protein"], 2)}* g\n'
        f'🍬 Carbonhydrates mass: *{round(nutrition_facts["carb"], 2)}* g\n'
        f'🧈 Fats mass: *{round(nutrition_facts["fat"], 2)}* g'
    )
    if is_saved == True:
        text+=(
            '\n✅ Saved to your meals'
        )
    
    return text.replace('.', '\.')


def build_inline_keyboard(is_saved: bool=False) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    if is_saved == False:
        builder.row(
                InlineKeyboardButton(text='Edit name', callback_data='name'),
                InlineKeyboardButton(text='Edit mass', callback_data='mass')  
        )
        builder.row(
            InlineKeyboardButton(text='Edit calories', callback_data='calories'),
            InlineKeyboardButton(text='Edit proteins', callback_data='protein')
        )
        builder.row(
            InlineKeyboardButton(text='Edit carbs', callback_data='carb'),
            InlineKeyboardButton(text='Edit fats', callback_data='fat')
        )
    
        builder.row(
            InlineKeyboardButton(text='Save to my meals', callback_data='Save to my meals')
        )
    else:
        builder.row()
    return builder.as_markup()


def build_reply_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text='🍽 Recognize nutrition'),
        KeyboardButton(text='📝 Edit My daily goal'),
        KeyboardButton(text='📊 Get today\'s statistics'),
    )
    builder.adjust(2)

    return builder.as_markup()

    
@form_router.message(
    F.text.endswith('Get today\'s statistics')
)
async def get_today_statistics(
    message: Message,
    state: FSMContext,
    session: AsyncSession
):

    user_id = int(message.from_user.id)
    
    print('start sql_get_user_todays_statistics')

    user_id = message.from_user.id

    daily_calories_goal = await sql_get_latest_daily_calories_goal(
        session=session, 
        user_id=user_id
    )

    statistics = await sql_get_user_todays_statistics(
        session=session, 
        user_id=user_id
    )
    print('finish sql_get_user_todays_statistics')
    print(f'daily_calories_goal: {daily_calories_goal}')
    print(f'daily_calories_goal: {statistics}')

    statistics = np.round(
        list(daily_calories_goal[0]) 
        + 
        list(statistics[0]), 
        1
    )

    (
        daily_calories_goal,
        total_calories,
        total_protein,
        total_carb,
        total_fat
    ) = statistics

    is_any_result_empty = any([x is None for x in statistics])

    if is_any_result_empty == True:
        await message.reply(
            text='For today there is no data'
        )
        return

    calories_percent = round(
        100 * (
            total_calories
            /
            daily_calories_goal
        )
    )

    # Caclulate progress
    progress_lenght = 20
    proportion_lenght = 10
    filled_block = '▓'
    empty_block = '░'

    percentage = round(
        min(
            progress_lenght, 
            (
                100 * (
                    total_calories
                    /
                    daily_calories_goal
                )
            ) 
            // 
            (
                100 // progress_lenght
            )
        )
    )

    calories_progress = (filled_block * percentage) + ((progress_lenght - percentage) * empty_block)
    
    normalize_nutrients_coefs = np.round(
        (
            100
            * 
            (
                np.array([total_protein, total_carb, total_fat])
                /
                max(total_protein, total_carb, total_fat) 
            )
        ) // (100 // proportion_lenght)
    ).astype(int)

    normalize_nutrients_coefs

    progresses = []

    for nutrient, proportion in zip(
        [total_protein, total_carb, total_fat],
        normalize_nutrients_coefs
    ):

        progresses.append(
            (filled_block * proportion) + ((proportion_lenght - proportion) * empty_block)
        )

    print('finish preparing stats')
    
    await message.reply(
        text=(
            '*Your today\'s calories statistics:*\n'
            f'📊 Calories consumed / goal: *{int(total_calories)}* / *{int(daily_calories_goal)}*\n' 
            f'{calories_progress} *{calories_percent}*%\n\n'
            '*Your today\'s nutrients proportion:*\n'
            f'{progresses[0]} 🍖 Protein *{round(total_protein, 1)}*g.\n' 
            f'{progresses[1]} 🍬 Carbs   *{round(total_carb, 1)}*g.\n'
            f'{progresses[2]} 🧈 Oils     *{round(total_fat, 1)}*g.' 
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_reply_keyboard()
    )


# @form_router.message(
#     F.text.endswith('Get today\'s statistics')
# )
# async def get_today_statistics_plot(
#     message: Message,
#     state: FSMContext,
#     session: AsyncSession
# ):
#     print('start sql_get_user_todays_statistics')

#     user_id = message.from_user.id
    
#     daily_calories_goal = await sql_get_latest_daily_calories_goal(
#         session=session, 
#         user_id=user_id
#     )

#     statistics = await sql_get_user_todays_statistics(
#         session=session, 
#         user_id=user_id
#     )
#     print('finish sql_get_user_todays_statistics')
#     print(f'daily_calories_goal: {daily_calories_goal}')
#     print(f'daily_calories_goal: {statistics}')

#     statistics = np.round(
#         list(daily_calories_goal[0]) 
#         + 
#         list(statistics[0]), 
#         1
#     )

#     (
#         daily_calories_goal,
#         total_calories,
#         total_protein,
#         total_carb,
#         total_fat
#     ) = statistics

#     is_any_result_empty = any([x is None for x in statistics])

#     if is_any_result_empty == True:
#         await message.reply(
#             text='For today there is no data'
#         )
#         return
    
#     await message.answer(
#         text=f'your today statistics: {statistics}'
#     )

#     # Offload the plotting function to a background thread
#     loop = asyncio.get_event_loop()
#     img_buf, fig = await loop.run_in_executor(
#         executor, 
#         lambda: today_statistic_plotter(
#             daily_calories_goal, total_calories, total_protein, total_carb, total_fat
#         )
#     )

#     # img_buf, fig = await today_statistic_plotter(
#     #     daily_calories_goal,
#     #     total_calories,
#     #     total_protein,
#     #     total_carb,
#     #     total_fat
#     # )

#     await message.answer_photo(
#         photo=BufferedInputFile(img_buf.read(), filename='daily_nutrition_plot.png'),
#         caption=(
#             'Your today\'s calories statistics:\n'
#             f'🧮 Calories consumed / goal: {int(total_calories)} / {int(daily_calories_goal)}\n'
#         ),
#         reply_markup=build_reply_keyboard()
#     )
#     img_buf.close()
#     plt.close(fig)
    
#     # await message.reply(
#     #     text=(
#     #         '*Your today\'s calories statistics:*\n'
#     #         f'🧮 Calories consumed / goal: *{int(total_calories)}* / *{int(daily_calories_goal)}*\n'
#     #     ),
#     #     photo=BufferedInputFile(img, filename='daily_nutrition_plot.png'),
#     #     parse_mode=ParseMode.MARKDOWN,
#     #     reply_markup=build_reply_keyboard()
#     # )



@form_router.message(
    F.text.endswith('Edit My daily goal')
    |
    F.text.endswith('/set_daily_goal')
)
async def edit_daily_goal_request(
    message: Message, 
    state: FSMContext,
    session
):
    await state.set_state(Form.edit_daily_goal)

    user_id = message.from_user.id

    latest_goal = await sql_get_latest_daily_calories_goal(
        session=session, 
        user_id=user_id
    )

    if latest_goal is not None:
        
        await message.answer(
            f'Your daily goal setted in: {latest_goal} kcall',
            reply_markup=build_reply_keyboard()
        )

    await message.answer(
        text='Please set amount of kcall that You want to consume daily\n',
        reply_markup=build_reply_keyboard()
    )


@form_router.message(Form.edit_daily_goal)
async def edit_daily_goal(
    message: Message, 
    state: FSMContext,
    session: AsyncSession
):
    daily_calories_goal = message.text
    
    try:
        daily_calories_goal = float(daily_calories_goal)
    except:        
        await message.answer(
            text='Amount of kcall to set as daily goal must be a number',
            reply_markup=build_reply_keyboard(),
        )
        await state.clear()
        return
    
    user_id = message.from_user.id

    first_name = message.from_user.first_name

    latest_goal = await sql_get_latest_daily_calories_goal(
        session=session, 
        user_id=user_id
    )

    if latest_goal is not None:
        
        if daily_calories_goal == float(latest_goal):
            await message.answer(
                f'Your daily goal setted in: {daily_calories_goal} kcall',
                reply_markup=build_reply_keyboard()
            )
            return
    
    first_name = message.from_user.first_name

    last_name = message.from_user.last_name

    user_name = message.from_user.username

    user_id = int(message.from_user.id)

    timestamp = datetime.datetime.now().astimezone().isoformat()

    chat_id = message.chat.id

    height, weight, age = None, None, None

    await sql_write_new_user(
        session=session, 
        user_row={
            'first_name': first_name,
            'last_name': last_name,
            'user_name': user_name,
            'user_id': user_id,
            'timestamp': timestamp,
            'chat_id': chat_id,
            'height': height,
            'weight': weight,
            'age': age,
            'daily_calories_goal': daily_calories_goal,
        }
    )

    await message.answer(
        f'Your daily goal setted in: {daily_calories_goal} kcall',
        reply_markup=build_reply_keyboard()
    )
    await state.clear()
        

@form_router.message(CommandStart())
async def welcome(
    message: Message,
    session: AsyncSession,
):
    first_name = message.from_user.first_name

    await message.answer(
        text=(
            f'👋 *Hey, {first_name}!* \n'
            'I\'m a helpful AI bot 🤖.'
            'Send me a photo 📸 of your food 🍽 \n'
            'and I\'ll recognize nutritional facts about it'
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_reply_keyboard()
    )

    await check_user_exist(message=message, session=session)


@form_router.message(F.text.endswith('Recognize nutrition'))
async def recognize_nutrition(
    message 
):
    await message.answer(
        text=(
            'Sure, send me a photo 📸 of your food 🍽 '
            'and I\'ll send recognized nutritional facts to you back:'
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_reply_keyboard(),
    )


@form_router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    # Send a response to the user
    await message.answer(
        "⚡️ Thank you for sending the photo! \n"
        "⚙️ It in processing, please wait your results",
        reply_markup=build_reply_keyboard()
    )

    await message.bot.send_chat_action(
        message.chat.id, 
        action=ChatAction.TYPING
    )

    # Get the largest available photo size
    photo_file_id = message.photo[-1].file_id
    
    # Download the photo as bytes
    bytes = io.BytesIO()
    photo_file = await message.bot.download(
        photo_file_id, 
        destination=bytes
    )

    img = Image.from_bytes(photo_file.read())

    request_parts = [PROMPT, img]
    
    response = model.generate_content(
        request_parts,
        generation_config=generation_config
    )
    result = response_to_dict(response)

    if result == 'no food' or result == 'not correct result':
        await message.answer(
            text=(
                '😔 Sorry I can not recognize food in your photo\n'
                '🙏 Please try once again'
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_reply_keyboard()
        )
    else:
        nutrition_facts = result

        text=text_from_nutrition_facts(nutrition_facts=nutrition_facts)
        
        await state.update_data(nutrition_facts=nutrition_facts)
        await state.update_data(chat_id=message.chat.id)
        await state.update_data(username=message.chat.username)
        await state.update_data(first_name=message.from_user.first_name)
        await state.update_data(last_name=message.from_user.last_name)
        await state.update_data(user_id=message.from_user.id)
        
        await message.answer(
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=build_inline_keyboard(),
        )
        await state.update_data(name=message.text)
#TODO rewrite build_inline_keyboard to show at first step only two buttons: save and edit


# Edit the results 
@form_router.callback_query(
    F.data.in_({
        'name',
        'calories',
        'mass',
        'fat',
        'carb',
        'protein' ,
    })
)
async def edit_data(callback_query: CallbackQuery, state: FSMContext):
    edit_request = callback_query.data
    data = await state.get_data()
    nutrition_facts = data['nutrition_facts']
    for key in nutrition_facts.keys():
        if key in edit_request:
            await state.update_data(key_to_edit=key)
            await state.set_state(Form.edit_request)
            await callback_query.message.answer(
                text=(
                    f'Current value of the {key} is: *{nutrition_facts[key]}* \n'
                    'Please send me a *correct* value of it'
                ),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=build_reply_keyboard()
            )


@form_router.message(Form.edit_request)
async def check_corrections(message: Message, state: FSMContext):
    data = await state.get_data()
    
    nutrition_facts = data['nutrition_facts']
    
    key_to_edit = data['key_to_edit']
    new_value = message.text
    
    if key_to_edit in [
        'calories', 'mass', 'fat', 'carb', 'protein'
    ]:
        try:
            new_value = float(new_value)
        except:

            await message.answer(
                text='Please enter a number'
            )

            return

    await state.update_data(new_value=new_value)

    builder = InlineKeyboardBuilder()

    builder.row(
        InlineKeyboardButton(
            text='Apply corrections', 
            callback_data='apply_corrections'
        ),
        InlineKeyboardButton(
            text=f'Save without corrections', 
            callback_data='Save to my meals'
        )  
    )

    reply_markup = builder.as_markup()

    await message.answer(
        text=(
            f'New value of the {key_to_edit} is: *{message.text}* \n'
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )


@form_router.callback_query(F.data == 'apply_corrections')
async def apply_corrections(callback_query: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    nutrition_facts = data['nutrition_facts']
    key_to_edit = data['key_to_edit']
    new_value = data['new_value']
    if key_to_edit in [
        'calories', 'mass', 'fat', 'carb', 'protein'
    ]:
        try:
            new_value = float(new_value)
        except:
            await callback_query.message.answer(
                text='Please enter a number'
            )
            
    nutrition_facts[key_to_edit] = data['new_value']
    await state.update_data(nutrition_facts=nutrition_facts)

    await callback_query.message.edit_text(
        text=text_from_nutrition_facts(
            nutrition_facts=nutrition_facts,
            is_saved=False
        ),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=build_inline_keyboard()
    )


@form_router.callback_query(F.data == 'Save to my meals')
async def write_nutrition_to_db(
    callback_query: CallbackQuery, 
    state: FSMContext,
    session: AsyncSession
):
    data = await state.get_data()
    
    nutrition_facts = data['nutrition_facts']
    timestamp = datetime.datetime.now().astimezone().isoformat()
    username = data['username']
    first_name = data['first_name']
    last_name = data['last_name']
    user_id = int(data['user_id'])
    chat_id = int(callback_query.message.chat.id)
    
    meal_row = {
        'timestamp': timestamp, 
        'user_id': user_id, 
        'dish_name': nutrition_facts['dish_name'],
        'calories': int(nutrition_facts['calories']),
        'mass': int(nutrition_facts['mass']),
        'protein': float(nutrition_facts['protein']),
        'carb': float(nutrition_facts['carb']),
        'fat': float(nutrition_facts['fat']),     
    }

    is_user_exist = await sql_check_if_user_exists(session=session, user_id=user_id)

    if is_user_exist == False:

        timestamp = datetime.datetime.now().astimezone().isoformat()

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_new_user(
            session=session, 
            user_row={
                'first_name': first_name,
                'last_name': last_name,
                'user_name': username,
                'user_id': user_id,
                'timestamp': timestamp,
                'chat_id': chat_id,
                'height': height,
                'weight': weight,
                'age': age,
                'daily_calories_goal': daily_calories_goal,
            }
        )

        await sql_write_nutrition(session=session, meal_row=meal_row)
    
    if is_user_exist == True:

        await sql_write_nutrition(session=session, meal_row=meal_row)

    await callback_query.message.edit_text(
        text=text_from_nutrition_facts(
            nutrition_facts=nutrition_facts,
            is_saved=True
        ),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=build_inline_keyboard(is_saved=True)
    )


BOT_SETTINGS_CHAPTER = None
#################################### Bot settings ####################################
# Middleware to inject session into handlers
class DataBaseSession(BaseMiddleware):
    def __init__(self, session_pool: async_sessionmaker):
        super().__init__()
        self.session_pool = session_pool

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        async with self.session_pool() as session:
            data['session'] = session  # Inject session into the data
            return await handler(event, data)


# Set the webhook for recieving updates in your url wia HTTPS POST with JSONs
async def on_webhook_startup(bot: Bot) -> None:
    # If you have a self-signed SSL certificate, then you will need to send a public
    # certificate to Telegram, for this case we'll use google cloud run service so
    # it not required to send sertificates
    await bot.set_webhook(
        f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}",
    )


async def on_startup(bot: Bot) -> None:
    
    await bot.delete_webhook()


# Close the engine when shutting down
async def on_shutdown(bot: Bot):
    await engine.dispose()


def webhook_main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()

    dp.include_router(form_router)

    # Register startup hook to initialize webhook
    dp.startup.register(on_webhook_startup)
    dp.shutdown.register(on_shutdown)

    dp.update.middleware(
        DataBaseSession(session_pool=session_maker)
    )

    # Initialize Bot instance with default bot properties 
    # which will be passed to all API calls
    bot = Bot(
        token=BOT_TOKEN, 
        default=DefaultBotProperties()
    )

    # Create aiohttp.web.Application instance
    app = web.Application()

    # Create an instance of request handler,
    # aiogram has few implementations for different cases of usage
    # In this example we use SimpleRequestHandler 
    # which is designed to handle simple cases
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )

    # Register webhook handler on application
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    # Mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # And finally start webserver
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


async def main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()

    dp.include_router(form_router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    dp.update.middleware(
        DataBaseSession(session_pool=session_maker)
    )

    # Initialize Bot instance with default bot properties 
    # which will be passed to all API calls
    bot = Bot(
        token=BOT_TOKEN, 
        default=DefaultBotProperties()
    )

    await dp.start_polling(bot)

    app = web.Application()
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # asyncio.run(main())
    webhook_main()
