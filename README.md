# 🤖 🥦 NutriBot: Telegram bot for Nutritional Analysis with Google AI
This Telegram bot, NutriBot, leverages Google AI model Gemini 1.5 Pro capabilities to analyze food images and provide nutritional information.
This bot:
* Serves with serverless GC Run thus automatically scallable by increasing amount of the requests
* Leverages GC Vertex with Gemini-1.5-pro-002 to apply nutrition recognition by photo of the food
* Has a PostgreSQL database to data processing

## 🌐 Live demo: https://t.me/nutritional_facts_bot

https://github.com/NikitiusIvanov/nutri-bot/blob/main/schemes/components.drawio.png

## ⚙️ Functionality:
Connects to the Telegram Bot API to receive user interactions.
Integrates with Google AI's LLM Gemini 1.5 Pro to analyze photos of food items.
Stores user data and recognized nutritional facts and send back reports with statistics.

## 🧠 Main Logic:
Welcome Users:

* Greet new users and register them in the system.
* Prompt users to send a photo of their food.
* Download and process the image.
* Send the image data to the Google AI model for analysis.
* Extract Nutrition. Interpret the AI model's response and extract recognized nutritional facts.
Handle cases where no food is detected or the analysis fails.
* Display the recognized nutritional information in a user-friendly format.
Offer options to edit or save the data for future reference.

### ✅ Additional Features:
* Allow users to set daily calorie goals.
* Provide statistics on consumed calories and macronutrients (protein, carbs, fat).

## 🫀 Technical Stack:
* aiogram - async wrapper telegram bot API
* google.generativeai - Google API for summarizing with Google's LLM model Gemini
* plotly - statistics visualisation
* Google Cloud Run - Serverless deployment
