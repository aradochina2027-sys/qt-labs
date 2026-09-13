import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from llm_service import LLMService

app = FastAPI(title="AI Support Assistant", version="1.0.0")

try:
    llm_service = LLMService()
except Exception:
    llm_service = None


class SupportRequest(BaseModel):
    message: str


class SupportResponse(BaseModel):
    answer: str
    execution_time_sec: float
    model_name: str


@app.post("/api/chat", response_model=SupportResponse)
async def handle_support_chat(payload: SupportRequest):
    if not llm_service:
        raise HTTPException(
            status_code=500,
            detail="Сервіс LLM не ініціалізовано. Перевірте змінні середовища у файлі .env",
        )

    user_text = payload.message.strip()

    if not user_text:
        raise HTTPException(
            status_code=400, detail="Звернення користувача не може бути порожнім."
        )

    try:
        answer, exec_time, model_name = llm_service.generate_response(user_text)
        return SupportResponse(
            answer=answer,
            execution_time_sec=exec_time,
            model_name=model_name,
        )
    except RuntimeError as err:
        err_msg = str(err)
        if "401" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "429" in err_msg:
            raise HTTPException(status_code=429, detail=err_msg)
        elif "504" in err_msg or "Timeout" in err_msg:
            raise HTTPException(status_code=504, detail=err_msg)
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Сервіс тимчасово недоступний: {err_msg}",
            )


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Помічник служби підтримки</title>
        <style>
            body {
                font-family: system-ui, -apple-system, sans-serif;
                max-width: 650px;
                margin: 40px auto;
                padding: 0 20px;
                background-color: #f9f9fb;
                color: #1a1a1a;
            }
            .container {
                background: #ffffff;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            h2 { margin-top: 0; color: #111827; }
            textarea {
                width: 100%;
                height: 100px;
                padding: 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                box-sizing: border-box;
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
            }
            textarea:focus { outline: none; border-color: #2563eb; }
            button {
                margin-top: 10px;
                padding: 10px 20px;
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                cursor: pointer;
            }
            button:hover { background-color: #1d4ed8; }
            button:disabled { background-color: #9ca3af; cursor: not-allowed; }
            .card {
                background: #f3f4f6;
                padding: 16px;
                border-radius: 8px;
                margin-top: 20px;
                white-space: pre-wrap;
            }
            .meta {
                font-size: 0.85em;
                color: #4b5563;
                margin-top: 12px;
                border-top: 1px solid #e5e7eb;
                padding-top: 8px;
            }
            .error {
                color: #991b1b;
                background: #fee2e2;
                padding: 12px;
                border-radius: 6px;
                margin-top: 20px;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Помічник служби підтримки</h2>
            <p style="font-size: 0.9em; color: #6b7280;">Введіть запитання щодо правил доставки, оплати, повернення або гарантії.</p>
            
            <textarea id="query" placeholder="Наприклад: Які умови повернення товару протягом 14 днів?"></textarea>
            <br>
            <button id="sendBtn" onclick="sendQuery()">Надіслати запит</button>

            <div id="result" style="display:none;" class="card">
                <div id="answerText"></div>
                <div class="meta">
                    Модель: <b id="modelName"></b> | Час виконання: <b id="execTime"></b> с
                </div>
            </div>
            <div id="errorText" style="display:none;" class="error"></div>
        </div>

        <script>
            async function sendQuery() {
                const queryInput = document.getElementById('query');
                const btn = document.getElementById('sendBtn');
                const resultDiv = document.getElementById('result');
                const errorDiv = document.getElementById('errorText');

                const queryText = queryInput.value.trim();
                if (!queryText) return;

                btn.disabled = true;
                btn.innerText = 'Обробка...';
                resultDiv.style.display = 'none';
                errorDiv.style.display = 'none';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: queryText })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Помилка виконання запиту');
                    }

                    document.getElementById('answerText').innerText = data.answer;
                    document.getElementById('modelName').innerText = data.model_name;
                    document.getElementById('execTime').innerText = data.execution_time_sec;
                    resultDiv.style.display = 'block';
                } catch (err) {
                    errorDiv.innerText = err.message;
                    errorDiv.style.display = 'block';
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Надіслати запит';
                }
            }
        </script>
    </body>
    </html>
    """