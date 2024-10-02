import asyncio
import io
import sys
import os
import datetime
import json
from typing import Any, Awaitable, Callable, Dict
import numpy as np
import pandas as pd
import logging
import google.generativeai as genai
import gspread
from vertexai.generative_models import GenerativeModel, Image
from aiohttp import web

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import CommandStart, Filter
from aiogram import F
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    BufferedInputFile,
    KeyboardButton,
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.utils.keyboard import (
    ReplyKeyboardBuilder,
    InlineKeyboardBuilder
)

from aiogram.client.session.aiohttp import AiohttpSession
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging
import asyncpg

# get db credentials
# Db Configuration
POSTGRES_USER=os.getenv('POSTGRES_USER')
POSTGRES_DB=os.getenv('POSTGRES_DB')
POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST=os.getenv('POSTGRES_HOST')

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "database": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD
}

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
genai.configure(api_key=GOOGLE_AI_API_KEY)
model = GenerativeModel("gemini-1.5-pro-preview-0514")
generation_config = {
    'temperature': 0,
}

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
        conn: asyncpg.connection.Connection, 
        user_id: int,
):
    """Checks if a user with the given user_id exists in the database.

    Args:
        con: An asyncpg connection.
        user_id: The user ID to check.

    Returns:
        True if the user exists, False otherwise.
    """

    exists = await conn.fetchrow(
        "SELECT EXISTS (SELECT 1 FROM users WHERE user_id = $1)", 
        user_id
    )

    return exists[0]


async def sql_get_latest_daily_calories_goal(
    conn: asyncpg.connection.Connection, 
    user_id: int,
):
    result = await conn.fetchrow(
        """
        SELECT daily_calories_goal
        FROM users
        WHERE user_id = $1
        ORDER BY timestamp DESC
        LIMIT 1
        """, 
        user_id
    )

    return result[0]


async def sql_write_new_user(
    conn: asyncpg.connection.Connection, 
    user_row: dict
):
    """
    """
    await conn.execute(
        """
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
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        """,
        user_row['first_name'],
        user_row['last_name'],
        user_row['user_name'],
        user_row['user_id'],
        user_row['chat_id'],
        user_row['height'],
        user_row['weight'],
        user_row['age'],
        user_row['daily_calories_goal']
    )


async def sql_write_nutrition(
    conn: asyncpg.connection.Connection, 
    meal_row: dict
):
    """
    """
    await conn.execute(
        """
        INSERT INTO meals (
            user_id,
            dish_name,
            calories,
            mass,
            protein,
            carb,
            fat 
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8
        )
        """,
        meal_row['user_id'],
        meal_row['dish_name'],
        meal_row['calories'],
        meal_row['mass'],
        meal_row['protein'],
        meal_row['carb'],
        meal_row['fat']
    )


async def sql_check_daily_goal_exists(
    conn: asyncpg.connection.Connection, 
    user_id: int,
) -> bool:
    query = """
    SELECT EXISTS (
        SELECT 1 FROM users 
        WHERE user_id = $1
        and daily_calories_goal is not null
    )
    """
    result = await conn.fetchrow(query, user_id)
    return result[0]


async def sql_get_user_todays_statistics(
    conn: asyncpg.connection.Connection, 
    user_id: int,
):
    query_daily_goal = (
        """
        SELECT daily_calories_goal
        FROM users
        WHERE user_id = $1
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )

    query_todays_statitics = (
        """
        SELECT 
            SUM(calories),
            SUM(protein),
            SUM(carb),
            SUM(fat)
        FROM meals
        WHERE timestamp::date = CURRENT_DATE
        AND user_id = $1
        GROUP BY user_id;
        """
    )
    
    daily_calories_goal_result = await conn.fetchrow(query_daily_goal, user_id)

    todays_statitics_result = await conn.fetchrow(query_todays_statitics, user_id)

    try: 
        (
            total_calories, 
            total_protein, 
            total_carb, 
            total_fat
        ) = todays_statitics_result

        daily_calories_goal = daily_calories_goal_result[0]

        return (
            daily_calories_goal, 
            total_calories, 
            total_protein, 
            total_carb, 
            total_fat
        )
    except:
        return None, None, None, None, None


async def check_user_exist(
    message: Message,
    conn: asyncpg.Connection
):

    is_user_exist = await sql_check_if_user_exists(conn, message.from_user.id)

    if is_user_exist == False:
        
        first_name = message.from_user.first_name

        last_name = message.from_user.last_name

        user_name = message.from_user.username

        user_id = message.from_user.id

        timestamp = datetime.datetime.now().astimezone().isoformat()

        chat_id = message.chat.id

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_new_user(
            conn=conn, 
            row_to_write={
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


async def response_to_dict(
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


async def today_statistic_plotter(
    daily_calories_goal,
    total_calories,
    total_protein,
    total_carb,
    total_fat
)-> go.Figure:
    
    fig = make_subplots(
        subplot_titles=[
            'Calories (kcal)', 
            'Macronutrients (g)'
        ],
        column_widths=[0.6, 0.4],
        rows=1, cols=2
    )

    fig.add_trace(
        go.Bar(
            x=['Calories'],
            y=[daily_calories_goal],
            marker=dict(color='rgba(116, 117, 116, 0.6)'),
            name='Daily goal',
            width=0.3
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=['Calories'],
            y=[total_calories],
            marker=dict(color='green'),
            name='Todays calories',
            width=0.295,
        ),
        row=1, col=1
    )

    for entity_name, color, value in zip(
        ['protein', 'carb', 'fat'],
        [
            'rgba(189, 88, 68, 1)',
            'rgba(139, 93, 212, 1)',
            'rgba(224, 169, 65, 1)'
        ],
        [
            total_protein,
            total_carb,
            total_fat
        ]
    ):
        fig.add_trace(
            go.Bar(
                x=[entity_name],
                y=[value],
                textfont=dict(size=15),
                marker=dict(color=color),
                name=entity_name,
            ),
            row=1,
            col=2
        )

    fig.update_layout(
        width=600,
        height=300,
        template='plotly_white',
        bargap=0.01,
        barmode='overlay',
        legend=dict(
            orientation='h', 
            x=0.5, 
            xanchor='center',
            borderwidth=0.01,
        ),
        margin = dict(t=50, l=0, r=0, b=0),
    )

    return fig


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
        KeyboardButton(text='📊 Get today\'s statisctics'),
    )
    builder.adjust(2)

    return builder.as_markup()

    
@form_router.message(
    F.text.endswith('Get today\'s statisctics')
)
async def get_today_statistics(
    message: Message, 
    state: FSMContext,
):
    data = await state.get_data()
    conn: asyncpg.Connection = data.get('conn')
    logging.debug(f'conn: {type(conn)}')

    await message.bot.send_chat_action(
        message.chat.id, 
        action=ChatAction.UPLOAD_DOCUMENT
    )

    user_id = str(message.from_user.id)

    results = await sql_get_user_todays_statistics(
        conn=conn, 
        user_id=user_id
    )

    is_any_result_empty = any([x is None for x in results])

    if not is_any_result_empty == False:
        datetime_now = (
            datetime
            .datetime.now()
            .astimezone()
            .isoformat()
            .split('.')[0]
        )

        (
            daily_calories_goal,
            total_calories,
            total_protein,
            total_carb,
            total_fat
        ) = results

        fig = await today_statistic_plotter(
            daily_calories_goal,
            total_calories,
            total_protein,
            total_carb,
            total_fat
        )

        output_buffer = io.BytesIO()
        
        fig.write_image(output_buffer, format="png")

        output_buffer.seek(0)

        file_bytes = output_buffer.read()

        document = BufferedInputFile(
            file=file_bytes, 
            filename=f'{datetime_now}_statistics.png'
        )

        await message.reply_photo(
            photo=document
        )

    if is_any_result_empty == True:
        await message.reply(
            text='Unfortunately there is no data'
        )


@form_router.message(
    F.text.endswith('Edit My daily goal')
    |
    F.text.endswith('/set_daily_goal')
)
async def edit_daily_goal_request(
    message: Message, 
    state: FSMContext
):
    await state.set_state(Form.edit_daily_goal)

    await message.answer(
        text='Please set amount of kcall that You want to consume daily\n'
    )


@form_router.message(Form.edit_daily_goal)
async def edit_daily_goal(
    message: Message, 
    state: FSMContext,
):
    data = await state.get_data()
    conn: asyncpg.Connection = data.get('conn')
    
    daily_calories_goal = message.text
    
    try:
        daily_calories_goal = float(daily_calories_goal)
    except:        
        await message.answer(
            text='Amount of kcall to set as daily goal must be a number',
            reply_markup=build_reply_keyboard(),
        )
        return
    
    user_id = message.from_user.id

    first_name = message.from_user.first_name

    latest_goal = await sql_get_latest_daily_calories_goal(
        conn=conn, 
        user_id=user_id
    )

    if daily_calories_goal == float(latest_goal):
        await message.answer(
            f'Your daily goal setted in: {daily_calories_goal} kcall',
            reply_markup=build_reply_keyboard()
        )
        return
    
    if daily_calories_goal == float(latest_goal):
        
        first_name = message.from_user.first_name

        last_name = message.from_user.last_name

        user_name = message.from_user.username

        user_id = message.from_user.id

        timestamp = datetime.datetime.now().astimezone().isoformat()

        chat_id = message.chat.id

        height, weight, age = None, None, None

        await sql_write_new_user(
            conn=conn, 
            row_to_write={
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
    state: FSMContext
):
    data = await state.get_data()
    conn: asyncpg.Connection = data.get('conn')

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

    await check_user_exist(message=message, conn=conn)

@form_router.message(F.text.endswith('Recognize nutrition'))
async def recognize_nutrition(
    message, 
):
    await message.answer(
        text=(
            'Sure, send me a photo 📸 of your food 🍽 '
            'and I\'ll send recognized nutritional facts to you back:'
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext):
    # Send a response to the user
    await message.answer(
        "⚡️ Thank you for sending the photo! \n"
        "⚙️ It in processing, please wait your results",
        reply_markup=ReplyKeyboardRemove()
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
        parse_mode=ParseMode.MARKDOWN
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
        
        reply_markup = build_inline_keyboard()
        
        await message.answer(
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
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
            await callback_query.message.answer(
                text=(
                    f'Current value of the {key} is: *{nutrition_facts[key]}* \n'
                    'Please send me a *correct* value of it'
                ),
                parse_mode=ParseMode.MARKDOWN
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
        InlineKeyboardButton(text='Apply corrections', callback_data='apply_corrections'),
        InlineKeyboardButton(text=f'Save without corrections', callback_data='Save to my meals')  
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
):
    data = await state.get_data()
    conn: asyncpg.Connection = data.get('conn')
    
    nutrition_facts = data['nutrition_facts']
    timestamp = datetime.datetime.now().astimezone().isoformat()
    username = data['username']
    first_name = data['first_name']
    last_name = data['last_name']
    user_id = data['user_id']
    chat_id = callback_query.message.chat.id
    meal_row = {
        'timestamp': timestamp, 
        'user_id': user_id, 
        'dish_name': nutrition_facts['dish_name'],
        'calories': nutrition_facts['calories'],
        'mass': nutrition_facts['mass'],
        'protein': nutrition_facts['protein'],
        'carb': nutrition_facts['carb'],
        'fat': nutrition_facts['fat'],     
    }

    is_user_exist = await sql_check_if_user_exists(conn=conn, user_id=user_id)

    if is_user_exist == False:

        timestamp = datetime.datetime.now().astimezone().isoformat()

        height, weight, age, daily_calories_goal = None, None, None, None

        await sql_write_new_user(
            conn=conn, 
            row_to_write={
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

        await sql_write_nutrition(conn=conn, meal_row=meal_row)
    
    if is_user_exist == True:

        await sql_write_nutrition(conn=conn, meal_row=meal_row)

    await callback_query.message.edit_text(
        text=text_from_nutrition_facts(
            nutrition_facts=nutrition_facts,
            is_saved=True
        ),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=None
    )

    await callback_query.message.answer(
        text='🆒 Your meal info succefully saved!',
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_reply_keyboard()
    )

BOT_SETTINGS_CHAPTER = None
#################################### Bot settings ####################################
# Middleware to add database connections 
# from connection pool
# in current handler data
class DatabaseMiddleware(BaseMiddleware):
    def __init__(self, pool: asyncpg.Pool):
        super().__init__()
        self.pool = pool

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        async with self.pool.acquire() as conn:
            data['conn'] = conn
            return await handler(event, data)


# Set the webhook for recieving updates in your url wia HTTPS POST with JSONs
async def on_startup(bot: Bot, dp: Dispatcher) -> None:
    pool = await asyncpg.create_pool(**DB_CONFIG)

    # Create the connection pool
    dp['pool'] = pool
    
    # Register middleware that add connection 
    # from pool into handler data
    dp.message.middleware(
        DatabaseMiddleware(pool)
    )
    
    # If you have a self-signed SSL certificate, then you will need to send a public
    # certificate to Telegram, for this case we'll use google cloud run service so
    # it not required to send sertificates
    await bot.set_webhook(
        f"{BASE_WEBHOOK_URL}{WEBHOOK_PATH}",
    )


async def on_shutdown(dp: Dispatcher):
    pool = dp.get('pool')
    if pool:
        await pool.close()


def main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()

    dp.include_router(form_router)

    # Initialize Bot instance with default bot properties 
    # which will be passed to all API calls
    bot = Bot(
        token=BOT_TOKEN, 
        default=DefaultBotProperties()
    )

    # Register startup hook to initialize webhook
    dp.startup.register(lambda: on_startup(bot, dp))
    
    dp.shutdown.register(lambda: on_shutdown(dp))

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
