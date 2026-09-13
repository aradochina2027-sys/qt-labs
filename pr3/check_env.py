import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

try:
    models = client.models.list()
    print("Доступні моделі для вашого ключа:")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"Помилка отримання списку моделей: {e}")