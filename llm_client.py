"""
LLM клиент для инвестиционного чат-бота
Обрабатывает запросы к ProxyAPI с обработкой ошибок
"""

import asyncio
import logging
from typing import Dict, List, Optional

import httpx

from config import config

logger = logging.getLogger(__name__)


class LLMAPIError(Exception):
    """Исключение для ошибок API LLM"""
    pass


class InvestmentLLMClient:
    """
    Клиент для работы с LLM через ProxyAPI
    Специализирован для задач инвестиционного консультирования
    """
    
    def __init__(self):
        self.base_url = config.api_base_url
        self.api_token = config.proxy_api_token
        self.model = config.model_name
        
        # HTTP клиент с таймаутом и retry
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=10)
        )
        
        logger.info(f"Инициализирован LLM клиент: {self.model}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    def _build_messages(self, user_query: str, context: Optional[str] = None) -> List[Dict]:
        """
        Формирует сообщения для chat completion API
        Специализированный промпт для инвестиционного консультирования
        """
        # Системный промпт для работы инвестиционного консультанта
        system_content = """Ты - профессиональный консультант по инвестиционным продуктам.

Твоя задача:
- Отвечать на вопросы клиентов об инвестиционных паях, ЗПИФ и инвестиционных услугах
- Использовать только предоставленную информацию из базы знаний
- Давать точные и профессиональные ответы
- Если информации нет в базе знаний - честно сказать об этом
- Не давать финансовых советов, только информацию о продуктах

Стиль: деловой, но понятный для клиентов."""

        messages = [{"role": "system", "content": system_content}]
        
        # Добавляем контекст если есть
        if context:
            user_content = f"""Информация из базы знаний:

{context}

Вопрос клиента: {user_query}

Ответь на основе предоставленной информации."""
        else:
            user_content = f"""Вопрос клиента: {user_query}

В базе знаний не найдено релевантной информации. Ответь что не можешь дать точный ответ и предложи обратиться к специалистам."""
            
        messages.append({"role": "user", "content": user_content})
        return messages
    
    async def generate_response(self, user_query: str, context: Optional[str] = None) -> str:
        """
        Генерирует ответ на вопрос пользователя
        Включает обработку ошибок и fallback сценарии
        """
        if not user_query or not user_query.strip():
            return "Пожалуйста, задайте вопрос об инвестиционных продуктах."
        
        # Проверяем конфигурацию
        if not self.api_token:
            logger.error("API токен не настроен")
            return "Сервис временно недоступен. Обратитесь к специалистам по телефону."
        
        try:
            messages = self._build_messages(user_query, context)
            
            # Подготавливаем запрос к API
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,  # Низкая креативность для точности
                "max_tokens": 800,
                "top_p": 0.9,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
                "User-Agent": "Investment-RAG-Bot/1.0"
            }
            
            logger.info(f"Отправка запроса к LLM: {len(user_query)} символов")
            
            # Делаем запрос с retry логикой
            for attempt in range(3):
                try:
                    response = await self.http_client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["choices"][0]["message"]["content"]
                        
                        logger.info(f"Получен ответ от LLM: {len(answer)} символов")
                        return answer.strip()
                        
                    elif response.status_code == 429:
                        # Rate limiting - ждем и повторяем
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit, ожидание {wait_time}с")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        error_detail = response.text[:200]
                        raise LLMAPIError(f"API error {response.status_code}: {error_detail}")
                        
                except httpx.TimeoutException:
                    if attempt == 2:  # Последняя попытка
                        raise LLMAPIError("Превышен таймаут запроса к LLM")
                    logger.warning(f"Таймаут на попытке {attempt + 1}")
                    await asyncio.sleep(1)
                    
                except httpx.RequestError as e:
                    if attempt == 2:  # Последняя попытка
                        raise LLMAPIError(f"Сетевая ошибка: {str(e)}")
                    logger.warning(f"Сетевая ошибка на попытке {attempt + 1}: {e}")
                    await asyncio.sleep(1)
            
            raise LLMAPIError("Исчерпаны попытки запроса к LLM")
            
        except LLMAPIError:
            # Пробрасываем наши ошибки как есть
            raise
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка в generate_response: {e}")
            raise LLMAPIError(f"Внутренняя ошибка: {str(e)}")
    
    async def health_check(self) -> bool:
        """Проверка доступности LLM API"""
        try:
            # Простой тестовый запрос
            test_response = await self.generate_response("Тест", "Тестовая информация")
            return bool(test_response)
        except Exception:
            return False


# Глобальный экземпляр клиента
llm_client = InvestmentLLMClient() 