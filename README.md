# 🤖 🥦 NutriBot: Telegram bot for Nutritional Analysis with Google AI
This Telegram bot, NutriBot, leverages the Google AI model Gemini 1.5 Pro's capabilities to analyze food images and provide nutritional information to help users track calories and nutrient consumptions.

This bot:  
* Serves with serverless GC Run, making it automatically scalable as the number of requests increases.  
* Leverages GC Vertex with Gemini-1.5-pro-002 to apply nutrition recognition from food photos.  
* Utilizes a PostgreSQL database for data processing.

## 🌐 Live demo: https://t.me/nutritional_facts_bot  
![Components Architecture](https://github.com/NikitiusIvanov/nutri-bot/blob/main/schemes/components.drawio.png)  

## 🧠 Main Logic:  
### Welcome Users:  
* Greet new users and register them in the system.  
* Prompt users to send a photo of their food.  
* Download and process the image.  
* Send the image data to the Google AI model for analysis.  
* Extract Nutrition: Interpret the AI model's response and extract recognized nutritional facts.  
* Handle cases where no food is detected or the analysis fails.  
* Display the recognized nutritional information in a user-friendly format.  
* Offer options to edit or save the data for future reference.  

### ✅ Additional Features:  
* Allows users to set daily calorie goals.  
* Allows editing of nutrition facts before storing them in the database.  
* Provides statistics on consumed calories and macronutrients (protein, carbs, fat).  

## 🫀 Technical Stack:  
* **Aiogram, SQLAlchemy, AIOHTTP** – for creating an asynchronous web server that performs concurrent request processing.  
* **GitHub Actions, Docker, Google Cloud Build** – for creating a simple yet convenient and functional automated cloud deployment pipeline.  
* **Google Cloud Run** – serverless computing for running bot instances and load balancing.  
* **Telegram Bot API with Webhook** – for redirecting user requests to the GC Run instance's URL.  
* **PostgreSQL** – database.  
* **Google Cloud Vertex with multimodal LLM Gemini-1.5-pro** – for AI tasks like image recognition and content generation.  
