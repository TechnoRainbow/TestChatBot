# RAG Чат-бот для инвестиционного консультирования

Интеллектуальный консультационный чат-бот на базе Retrieval-Augmented Generation (RAG) для инвестиционных вопросов.

## 🎯 Описание

Система предоставляет консультации по инвестиционным продуктам, используя векторный поиск в базе знаний и генерацию ответов через LLM. 

## 🛠 Технологический стек

- **Backend**: FastAPI
- **LLM**: ProxyAPI (OpenAI GPT-3.5-turbo)
- **Векторная БД**: FAISS (Facebook AI Similarity Search)
- **Эмбеддинги**: sentence-transformers (multilingual)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## 📦 Установка и запуск

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск приложения
```bash
python app.py
```

Приложение будет доступно по адресу: http://localhost:8000

## 🔧 Структура проекта

```
├── app.py              # FastAPI приложение с RAG пайплайном
├── config.py           # Конфигурация системы
├── llm_client.py       # Клиент для работы с LLM API
├── knowledge_base.py   # Векторная база знаний с FAISS
├── requirements.txt    # Зависимости проекта
└── README.md          # Документация
```

## 🏗 Архитектура RAG-пайплайна

### Компоненты системы:

1. **Knowledge Base (FAISS)**
   - Векторизация документов с помощью sentence-transformers
   - Индексирование в FAISS для быстрого поиска
   - Поиск по косинусному сходству

2. **LLM Client (ProxyAPI)**
   - Интеграция с OpenAI GPT-3.5-turbo через ProxyAPI
   - Retry логика и обработка ошибок
   - Контекстное формирование промптов

3. **FastAPI Application**
   - REST API для чат-бота
   - Веб-интерфейс для пользователей
   - Мониторинг и health checks

### Поток обработки запроса:

```
Пользователь → FastAPI → Knowledge Base (FAISS) → LLM Client → Ответ
                ↓              ↓                    ↓
            Валидация    Векторный поиск    Генерация ответа
```

## 🧪 Примеры запросов

Примеры запросов для тестирования:

1. **"Что такое инвестиционный пай (Пай)?"**
   - Ожидаемый ответ: Определение, права пайщика, учёт в реестре

2. **"Перед покупкой паев нужно пройти тестирование. Что это значит?"**
   - Ожидаемый ответ: Процедура тестирования, 7 вопросов, возможность пересдачи

3. **"Возможно ли вернуть средства из ЗПИФ до прекращения фонда?"**
   - Ожидаемый ответ: Продажа на вторичном рынке, различия для типов инвесторов

## 🚀 API Документация

### POST /chat
Основной эндпоинт для обработки запросов:

**Запрос:**
```json
{
  "query": "Что такое инвестиционный пай?"
}
```

**Ответ:**
```json
{
  "response": "Инвестиционный пай — это именная ценная бумага...",
  "context_found": true,
  "processing_time": 2.5
}
```

### GET /health
Проверка состояния системы:
```json
{
  "status": "healthy",
  "components": {
    "knowledge_base": {"status": "healthy", "documents": 16},
    "llm_client": {"status": "healthy", "model": "gpt-3.5-turbo"}
  }
}
```

### GET /stats
Статистика системы для мониторинга.

## 📊 Мониторинг

Базовые метрики для отслеживания:
- Время ответа системы
- Успешность поиска контекста
- Доступность LLM API
- Количество запросов в единицу времени

## 📈 Рекомендации по улучшению

### Краткосрочные улучшения:
1. **Расширение базы знаний** - добавление большего количества документов
2. **Улучшение чанкинга** - разбиение длинных документов на смысловые части
3. **Настройка гиперпараметров** - экспериментирование с порогами сходства
4. **Кэширование** - Redis для часто запрашиваемых ответов

### Среднесрочные улучшения:
1. **Hybrid поиск** - комбинирование векторного и ключевого поиска
2. **Reranking** - переранжирование результатов поиска
3. **Мультимодальность** - поддержка документов, изображений
4. **A/B тестирование** - сравнение разных LLM и embedding моделей

### Долгосрочные улучшения:
1. **Fine-tuning** LLM на доменной специфике инвестиций
2. **Продвинутый RAG** - GraphRAG, агентный подход
3. **Персонализация** - адаптация ответов под тип клиента
4. **Интеграция с внешними API** - актуальные данные о рынке

## 🔧 Конфигурация

Основные параметры настраиваются через переменные окружения:
- `PROXYAPI_TOKEN` - токен для доступа к LLM
- `LLM_PROVIDER` - провайдер LLM (по умолчанию: openai)
- `MODEL_NAME` - модель LLM (по умолчанию: gpt-3.5-turbo)
- `HOST`, `PORT` - параметры сервера

---
