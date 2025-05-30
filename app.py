"""
RAG чат-бот для инвестиционного консультирования
FastAPI приложение с поддержкой FAISS и обработкой ошибок
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

from llm_client import llm_client, LLMAPIError
from knowledge_base import knowledge_base
from config import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Investment RAG Чат-бот",
    description="Консультационный чат-бот по инвестиционным продуктам",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    """Модель запроса к чат-боту"""
    query: str = Field(..., min_length=1, max_length=1000, description="Вопрос пользователя")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Вопрос не может быть пустым')
        return v.strip()


class ChatResponse(BaseModel):
    """Модель ответа чат-бота"""
    response: str = Field(..., description="Ответ на вопрос")
    context_found: bool = Field(..., description="Найден ли контекст в базе знаний")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Главная страница с интерфейсом чат-бота"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Инвестиционный помощник</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #f0f0f0;
            }
            
            .header h1 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.2em;
            }
            
            .header p {
                color: #7f8c8d;
                font-size: 1.1em;
            }
            
            .chat-container {
                height: 450px;
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                overflow-y: auto;
                background: #fafbfc;
            }
            
            .message {
                margin: 15px 0;
                padding: 12px 18px;
                border-radius: 10px;
                max-width: 85%;
                word-wrap: break-word;
                line-height: 1.4;
            }
            
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: #ecf0f1;
                color: #2c3e50;
                border-left: 4px solid #3498db;
            }
            
            .input-section {
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .message-input {
                flex: 1;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            
            .message-input:focus {
                outline: none;
                border-color: #3498db;
            }
            
            .send-button {
                padding: 15px 25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: transform 0.2s;
            }
            
            .send-button:hover:not(:disabled) {
                transform: translateY(-2px);
            }
            
            .send-button:disabled {
                background: #95a5a6;
                cursor: not-allowed;
                transform: none;
            }
            
            .examples {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #e9ecef;
            }
            
            .examples h3 {
                margin-bottom: 15px;
                color: #2c3e50;
            }
            
            .example-btn {
                display: block;
                width: 100%;
                margin: 8px 0;
                padding: 12px 15px;
                background: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
                color: #2c3e50;
                transition: all 0.3s;
                text-align: left;
            }
            
            .example-btn:hover {
                background: #3498db;
                color: white;
                border-color: #3498db;
                transform: translateX(5px);
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin: 15px 0;
            }
            
            .status {
                text-align: center;
                margin-top: 10px;
                font-size: 12px;
                color: #95a5a6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>💼 Инвестиционный помощник</h1>
                <p>Консультационный чат-бот по инвестиционным продуктам</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    <strong>Консультант:</strong> Добро пожаловать! Я помогу ответить на ваши вопросы об инвестиционных паях, ЗПИФ и других инвестиционных продуктах.
                </div>
            </div>
            
            <div class="loading" id="loading">Поиск информации в базе знаний...</div>
            
            <div class="input-section">
                <input 
                    type="text" 
                    id="messageInput" 
                    class="message-input" 
                    placeholder="Ваш вопрос об инвестиционных продуктах..."
                    maxlength="1000"
                />
                <button id="sendButton" class="send-button" onclick="sendMessage()">
                    Отправить
                </button>
            </div>
            
            <div class="examples">
                <h3>📋 Примеры вопросов:</h3>
                <a href="#" class="example-btn" onclick="askExample('Что такое инвестиционный пай (Пай)?')">
                    🔹 Что такое инвестиционный пай (Пай)?
                </a>
                <a href="#" class="example-btn" onclick="askExample('Перед покупкой паев нужно пройти тестирование. Что это значит?')">
                    🔹 Перед покупкой паев нужно пройти тестирование. Что это значит?
                </a>
                <a href="#" class="example-btn" onclick="askExample('Возможно ли вернуть средства из ЗПИФ до прекращения фонда?')">
                    🔹 Возможно ли вернуть средства из ЗПИФ до прекращения фонда?
                </a>
                <a href="#" class="example-btn" onclick="askExample('Какие риски у инвестирования в ЗПИФ?')">
                    🔹 Какие риски у инвестирования в ЗПИФ?
                </a>
            </div>
            
            <div class="status">
                RAG система на базе FAISS
            </div>
        </div>

        <script>
            function addMessage(text, isUser = false, hasContext = null) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
                
                let content = text;
                if (!isUser) {
                    content = '<strong>Консультант:</strong> ' + text;
                    if (hasContext !== null) {
                        const contextIcon = hasContext ? '📚' : '❓';
                        content += `<br><small style="opacity: 0.7;">${contextIcon} ${hasContext ? 'Информация найдена в базе знаний' : 'Информация не найдена в базе знаний'}</small>`;
                    }
                }
                
                messageDiv.innerHTML = content;
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }
            
            function setLoading(isLoading) {
                const loading = document.getElementById('loading');
                const sendButton = document.getElementById('sendButton');
                const messageInput = document.getElementById('messageInput');
                
                loading.style.display = isLoading ? 'block' : 'none';
                sendButton.disabled = isLoading;
                messageInput.disabled = isLoading;
                sendButton.textContent = isLoading ? 'Обрабатываю...' : 'Отправить';
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                addMessage(message, true);
                input.value = '';
                setLoading(true);
                
                try {
                    const startTime = Date.now();
                    
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: message })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        const duration = Date.now() - startTime;
                        addMessage(result.response, false, result.context_found);
                        console.log(`Ответ получен за ${duration}мс, контекст: ${result.context_found}`);
                    } else {
                        addMessage(`Ошибка: ${result.detail || 'Неизвестная ошибка'}`, false);
                    }
                    
                } catch (error) {
                    console.error('Ошибка запроса:', error);
                    addMessage('Извините, произошла ошибка при обработке запроса. Попробуйте позже.', false);
                } finally {
                    setLoading(false);
                }
            }
            
            function askExample(question) {
                document.getElementById('messageInput').value = question;
                sendMessage();
            }
            
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            document.getElementById('messageInput').focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/chat", response_model=ChatResponse)
async def process_chat_request(request: ChatRequest):
    """
    Обработка запроса чат-бота с полным RAG пайплайном
    1. Поиск релевантного контекста в FAISS
    2. Генерация ответа через LLM с контекстом
    """
    start_time = time.time()
    
    try:
        logger.info(f"Получен запрос: '{request.query[:50]}...'")
        
        # Этап 1: Поиск в векторной базе знаний
        context = knowledge_base.search(request.query, top_k=3)
        context_found = bool(context)
        
        logger.info(f"Поиск в базе знаний: {'найден контекст' if context_found else 'контекст не найден'}")
        
        # Этап 2: Генерация ответа через LLM
        try:
            response_text = await llm_client.generate_response(
                user_query=request.query,
                context=context
            )
        except LLMAPIError as e:
            logger.error(f"Ошибка LLM API: {e}")
            # Fallback ответ при ошибке LLM
            if context_found:
                response_text = f"Извините, сервис генерации ответов временно недоступен. Вот информация из нашей базы знаний:\n\n{context}"
            else:
                response_text = "Извините, сервис временно недоступен. Пожалуйста, обратитесь к специалистам по телефону или через сайт."
        
        processing_time = time.time() - start_time
        
        logger.info(f"Запрос обработан за {processing_time:.2f}с")
        
        return ChatResponse(
            response=response_text,
            context_found=context_found,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Неожиданная ошибка в чат-боте: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера. Попробуйте позже."
        )


@app.get("/health")
async def health_check():
    """Расширенная проверка здоровья системы"""
    try:
        # Проверяем компоненты системы
        kb_stats = knowledge_base.get_stats()
        llm_health = await llm_client.health_check()
        
        return {
            "status": "healthy" if llm_health else "degraded",
            "components": {
                "knowledge_base": {
                    "status": "healthy",
                    "documents": kb_stats["total_documents"],
                    "embedding_dimension": kb_stats["embedding_dimension"]
                },
                "llm_client": {
                    "status": "healthy" if llm_health else "error",
                    "model": config.model_name
                }
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/stats")
async def get_system_stats():
    """Статистика системы для мониторинга"""
    try:
        kb_stats = knowledge_base.get_stats()
        return {
            "knowledge_base": kb_stats,
            "config": {
                "model": config.model_name,
                "provider": config.llm_provider
            }
        }
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения статистики")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Запуск RAG чат-бота...")
    logger.info(f"LLM модель: {config.model_name}")
    logger.info(f"Провайдер: {config.llm_provider}")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info"
    ) 