import io
import sys
import os
import datetime
from typing import Any, Awaitable, Callable, Dict
import numpy as np
import logging
import matplotlib.pyplot as plt 

import asyncio
from aiohttp import web

# Google Generative AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Image

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker, 
)

from sqlalchemy import sql

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
)
from aiogram.utils.keyboard import (
    ReplyKeyboardBuilder,
    InlineKeyboardBuilder
)
from aiogram.client.session.aiohttp import AiohttpSession

####################### GLOBAL_INITS_CHAPTER #######################
GLOBAL_INITS_CHAPTER = None
# Settings for logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
try:
    is_local_debug = bool(os.getenv('IS_LOCAL_DEBUG'))
except:
    is_local_debug = False
logging.info(f'IS_LOCAL_DEBUG: {is_local_debug}')

# Bot token can be obtained via https://t.me/BotFather
bot_token = os.getenv('BOT_TOKEN')

# Your CG Run instance url
base_webhook_url = os.getenv('BASE_WEBHOOK_URL')

# Path to webhook route, on which Telegram will send requests
webhook_path = "/webhook"

# Webserver settings
web_server_host = "0.0.0.0"

# Port for incoming request
web_server_port = 8080

def init_vertex_gemini(model_name: str) -> GenerativeModel | None:
    """Getting env variables for vertexai initialization
    * assume existing varialbes in env:
        * PROJECT_ID # your CG project ID
        * REGION # your GC project region
    
    * if env variable PROJECT_ID is None: return None
    (exclusion for performing unit tests on github Pytest)
    """
    PROJECT_ID = os.getenv('PROJECT_ID')
    REGION = os.getenv('REGION')

    if PROJECT_ID is not None: 
        vertexai.init(project=PROJECT_ID, location=REGION)

        generation_config = {
            'temperature': 0,
        }

        model = GenerativeModel(
            model_name=model_name, 
            generation_config=generation_config
        )
    else:
        model = None

    return model


def create_asyncpg_session_pool() -> AsyncSession:
    """
    * creates an async session with pool of the connections
    for further registering in a Middleware for injection
    connections into handlers
    * assume there is in env variables:
        * POSTGRES_USER
        * POSTGRES_DB
        * POSTGRES_PASSWORD
        * POSTGRES_HOST
    """
    # get db credentials
    # Db Configuration
    POSTGRES_USER=os.getenv('POSTGRES_USER')
    POSTGRES_DB=os.getenv('POSTGRES_DB')
    POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD')
    POSTGRES_HOST=os.getenv('POSTGRES_HOST')
    POSTGRES_PORT='5432'

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

    session_pool = async_sessionmaker(
        bind=engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )

    return session_pool


# Iinit with "gemini-1.5-pro-002"
model = init_vertex_gemini(model_name="gemini-1.5-pro-002")

session_pool = create_asyncpg_session_pool()

# Prompt for food nutrition recognition by photo
prompt = """
You are a helpful AI assistant that helps people collect data about their diets
by food photos that they send to you.
Recognize food in this picture and give your estimation of the:
* total amount of calories in kcal,
* mass in grams,
* fat in grams,
* carbohydrates in grams


If you recognize there is no food in the photo
write your answer by following format and nothing more:


no food


If you recognize some food on the photo use low-high borders
for you estimation of the nutritional facts
and write your answer by following format and nothing more:


dish_name: apple pie
calories: 230, 240
mass: 340, 350
fat: 5.0, 5.5
carb: 22, 25
protein: 24, 25
"""

####################### DATA_PROCESSING_CHAPTER #######################
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
        sql.text(
            """
            SELECT EXISTS (
                SELECT 1 FROM users 
                WHERE user_id = :user_id
            )
            """
        ), 
        {'user_id': user_id}
    )
    result = result.fetchone()

    return result[0]


async def sql_write_user(
    session: AsyncSession, 
    user_row: dict
):
    """
    """
    await session.execute(
        sql.text("""
        INSERT INTO users (
            first_name,
            user_name,
            user_id,
            height,
            weight,
            age,
            daily_calories_goal
        ) VALUES (
            :first_name,
            :user_name,
            :user_id,
            :height,
            :weight,
            :age,
            :daily_calories_goal
       )
        """),
        {
            'first_name': user_row.get('first_name'),
            'user_name': user_row.get('user_name'),
            'user_id': user_row.get('user_id'),
            'height': user_row.get('height'),
            'weight': user_row.get('weight'),
            'age': user_row.get('age'),
            'daily_calories_goal': user_row.get('daily_calories_goal')
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
        sql.text("""
        INSERT INTO meals (
            user_id,
            dish_name,
            calories,
            mass,
            protein,
            carb,
            fat,
            photo_file_id
        ) VALUES (
            :user_id,
            :dish_name,
            :calories,
            :mass,
            :protein,
            :carb,
            :fat,
            :photo_file_id
        )
        """),
        {
            'user_id': meal_row.get('user_id'),
            'dish_name': meal_row.get('dish_name'),
            'calories': meal_row.get('calories'),
            'mass': meal_row.get('mass'),
            'protein': meal_row.get('protein'),
            'carb': meal_row.get('carb'),
            'fat': meal_row.get('fat'),
            'photo_file_id': meal_row.get('photo_file_id')
        }
    )

    await session.commit()


async def sql_check_daily_goal_exists(
    session: AsyncSession, 
    user_id: int,
) -> bool:
    query = sql.text("""
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
    
    result = await session.execute(
        sql.text("""
        SELECT daily_calories_goal
        FROM users
        WHERE user_id = :user_id
        ORDER BY timestamp DESC
        LIMIT 1
        """), 
        {'user_id': user_id}
    )

    return result.fetchall()


async def sql_get_user_todays_statistics(
    session: AsyncSession,
    user_id: int,
):

    query_todays_nutrition = sql.text(
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


async def write_user_if_not_exist(
    message: Message,
    session: AsyncSession,
):

    is_user_exist = await sql_check_if_user_exists(session, message.from_user.id)

    if is_user_exist == False:
        
        first_name = message.from_user.first_name

        user_name = message.from_user.username

        user_id = message.from_user.id

        timestamp = datetime.datetime.now().astimezone().isoformat()

        chat_id = message.chat.id

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_user(
            session=session, 
            user_row={
                'first_name': first_name,
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

#TODO need to solve problem with delaying 
# and switch to send stats with plots
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


####################### BOT_UTILITES_CHAPTER #######################
BOT_UTILITES_CHAPTER = None

# Create FSM form for storing data 
# and perform logic chains scenarios
class Form(StatesGroup):
    chat_id = State()
    photo_ok = State()
    photo_file_id = State()
    nutrition_ok = State()
    nutrition_facts = State()
    username = State()
    first_name = State()
    user_id = State()
    edit_request = State()
    key_to_edit = State()
    new_value = State()
    edit_daily_goal = State()
    statistics = State()


form_router = Router()

def inline_keyboard_recognized(is_saved: bool=False) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    if is_saved == False:
        builder.row(
            InlineKeyboardButton(text='Save to my meals', callback_data='Save to my meals'),
            InlineKeyboardButton(text='Edit nutrition facts', callback_data='Edit nutrition')  
        )
    else:
        builder.row()
    return builder.as_markup()


def inline_keyboard_in_edition(is_saved: bool=False) -> InlineKeyboardMarkup:
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


def reply_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text='🍽 Recognize nutrition'),
        KeyboardButton(text='📝 Set or Edit My daily goal'),
        KeyboardButton(text='📊 Get today\'s statistics'),
    )
    builder.adjust(2)

    return builder.as_markup()

####################### BOT_HANDLERS_CHAPTER #######################
BOT_HANDLERS_CHAPTER = None 

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

    daily_calories_goal = await sql_get_daily_goal(
        session=session, 
        user_id=user_id
    )
    daily_calories_goal = daily_calories_goal[0][0]
    
    if daily_calories_goal is None:
        await message.reply(
            text='Please set daily calories goal'
        )
        return
    else:
        daily_calories_goal = round(daily_calories_goal, 1)

    statistics = await sql_get_user_todays_statistics(
        session=session, 
        user_id=user_id
    )
    print('finish sql_get_user_todays_statistics')
    print(f'daily_calories_goal: {daily_calories_goal}')
    print(f'daily_calories_goal: {statistics}')

    try:
        statistics = np.round(
            list(statistics[0]),
            1
        )
    except:
        await message.reply(
            text='For today there is no data'
        )
        return

    (
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
        reply_markup=reply_keyboard()
    )



@form_router.message(
    F.text.endswith('Set or Edit My daily goal')
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

    latest_goal = await sql_get_daily_goal(
        session=session, 
        user_id=user_id
    )

    latest_goal = latest_goal[0][0]

    if latest_goal is not None:
        
        await message.answer(
            f'Your daily goal setted in: {latest_goal} kcall',
            reply_markup=reply_keyboard()
        )

    await message.answer(
        text='Please set amount of kcall that You want to consume daily\n',
        reply_markup=reply_keyboard()
    )


@form_router.message(Form.edit_daily_goal)
async def edit_daily_goal(
    message: Message, 
    state: FSMContext,
    session: AsyncSession
):
    daily_calories_goal = message.text
    
    try:
        daily_calories_goal = int(float(daily_calories_goal))
    except:        
        await message.answer(
            text='Amount of kcall to set as daily goal must be a number',
            reply_markup=reply_keyboard(),
        )
        await state.clear()
        return
    
    user_id = message.from_user.id

    first_name = message.from_user.first_name

    latest_goal = await sql_get_daily_goal(
        session=session, 
        user_id=user_id
    )

    latest_goal = latest_goal[0][0]

    if latest_goal is not None:
        
        if daily_calories_goal == int(float(latest_goal)):
            await message.answer(
                f'Your daily goal setted in: {daily_calories_goal} kcall',
                reply_markup=reply_keyboard()
            )
            return
    
    first_name = message.from_user.first_name

    user_name = message.from_user.username

    user_id = int(message.from_user.id)

    timestamp = datetime.datetime.now().astimezone().isoformat()

    chat_id = message.chat.id

    height, weight, age = None, None, None

    await sql_write_user(
        session=session, 
        user_row={
            'first_name': first_name,
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
        reply_markup=reply_keyboard()
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
        reply_markup=reply_keyboard()
    )

    await write_user_if_not_exist(message=message, session=session)


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
        reply_markup=reply_keyboard(),
    )


@form_router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    # Send a response to the user
    await message.answer(
        "⚡️ Thank you for sending the photo! \n"
        "⚙️ It in processing, please wait your results",
        reply_markup=reply_keyboard()
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

    request_parts = [prompt, img]
    
    response = model.generate_content(
        request_parts
    )
    result = response_to_dict(response)

    if result == 'no food' or result == 'not correct result':
        await message.answer(
            text=(
                '😔 Sorry I can not recognize food in your photo\n'
                '🙏 Please try once again'
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_keyboard()
        )
    else:
        nutrition_facts = result

        text=text_from_nutrition_facts(nutrition_facts=nutrition_facts)
        
        await state.update_data(photo_file_id=photo_file_id)
        await state.update_data(nutrition_facts=nutrition_facts)
        await state.update_data(chat_id=message.chat.id)
        await state.update_data(username=message.chat.username)
        await state.update_data(first_name=message.from_user.first_name)
        await state.update_data(user_id=message.from_user.id)
        
        await message.answer(
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=inline_keyboard_recognized(),
        )
        await state.update_data(name=message.text)


@form_router.callback_query(
    F.data == 'Edit nutrition'
)
async def edit_data_change_inline(
    callback_query: CallbackQuery, 
    state: FSMContext
):
    await callback_query.message.edit_text(
        text=callback_query.message.text.replace('.', '\.'),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=inline_keyboard_in_edition()
    )

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
            await callback_query.message.edit_reply_markup(
                text=callback_query.message.message_id,
                reply_markup=None,
            )
            await callback_query.message.answer(
                text=(
                    f'Current value of the {key} is: *{nutrition_facts[key]}* \n'
                    'Please send me a *correct* value of it'
                ),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_keyboard()
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
        reply_markup=inline_keyboard_recognized()
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
    username = data.get('username')
    first_name = data.get('first_name')
    user_id = int(data.get('user_id'))
    photo_file_id = data.get('photo_file_id')
    
    meal_row = {
        'timestamp': timestamp, 
        'user_id': user_id,
        'dish_name': nutrition_facts['dish_name'],
        'calories': int(nutrition_facts['calories']),
        'mass': int(nutrition_facts['mass']),
        'protein': float(nutrition_facts['protein']),
        'carb': float(nutrition_facts['carb']),
        'fat': float(nutrition_facts['fat']), 
        'photo_file_id': photo_file_id,
    }

    is_user_exist = await sql_check_if_user_exists(session=session, user_id=user_id)

    if is_user_exist == False:

        timestamp = datetime.datetime.now().astimezone().isoformat()

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_user(
            session=session, 
            user_row={
                'first_name': first_name,
                'user_name': username,
                'user_id': user_id,
                'timestamp': timestamp,
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
        reply_markup=inline_keyboard_recognized(is_saved=True)
    )


#################################### BOT_SETTINGS_CHAPTER ####################################
BOT_SETTINGS_CHAPTER = None

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
    """does: await bot.set_webhook(
        f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}",
    )"""

    await bot.set_webhook(
        f"{base_webhook_url}{webhook_path}",

    )


# Remove webhook if it setted
async def on_pooling_startup(bot: Bot) -> None:
    """does: await bot.delete_webhook()"""
    await bot.delete_webhook()


def webhook_main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()

    dp.include_router(form_router)

    # Register startup hook to initialize webhook
    dp.startup.register(on_webhook_startup)

    dp.update.middleware(
        DataBaseSession(session_pool=session_pool)
    )

    # Initialize Bot instance with default bot properties 
    # which will be passed to all API calls
    bot = Bot(
        token=bot_token, 
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
    webhook_requests_handler.register(app, path=webhook_path)

    # Mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # And finally start webserver
    web.run_app(app, host=web_server_host, port=web_server_port)


async def pooling_main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()

    dp.include_router(form_router)
    dp.startup.register(on_pooling_startup)

    dp.update.middleware(
        DataBaseSession(session_pool=session_pool)
    )

    # Initialize Bot instance with default bot properties 
    # which will be passed to all API calls
    bot = Bot(
        token=bot_token, 
        default=DefaultBotProperties()
    )

    await dp.start_polling(bot)

    app = web.Application()
    web.run_app(app, host=web_server_host, port=web_server_port)


if __name__ == "__main__":
    if is_local_debug:
        asyncio.run(pooling_main())
    else:
        webhook_main()
