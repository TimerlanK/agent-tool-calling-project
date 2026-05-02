# Киноманьяк: AI movie agent

Локальный AI-агент эксперт по фильмам на **Ollama + LangChain LCEL + bind_tools**.
Данные о фильмах агент проверяет через OMDb API.

## Возможности

- Поиск фильма по названию.
- Поиск списка фильмов.
- Сравнение двух фильмов.
- Фильтрация по жанру.
- Поиск фильмов выше заданного IMDb рейтинга.
- Multi-tool calling: агент может вызывать 2, 3 и больше инструментов в одном запросе.

## Установка

1. Установите Ollama: <https://ollama.com/>
2. Скачайте модель с tool/function calling:

```bash
ollama pull qwen2.5:7b-instruct
```

3. Создайте окружение и установите зависимости:

На Ubuntu/Debian может понадобиться:

```bash
sudo apt install python3-venv python3-pip
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Получите ключ OMDb на <http://www.omdbapi.com/apikey.aspx>.

5. Создайте `.env`:

```bash
cp .env.example .env
```

Заполните:

```bash
OMDB_API_KEY=your_omdb_api_key
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
```

## Запуск

Интерактивный режим:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli
```

Один вопрос:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli "Сравни рейтинги Matrix и Inception"
```

С выводом вызванных tools:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --trace "Сравни рейтинги Matrix и Inception"
```

## Примеры multi-tool запросов

```text
Сравни рейтинги Matrix и Inception
```

Ожидаемо: агент вызывает инструмент сравнения или получает данные по обоим фильмам, затем сравнивает.

```text
Найди фильмы Batman, оставь только триллеры и покажи лучшие по IMDb рейтингу
```

Ожидаемо: агент ищет список, получает детали, фильтрует по жанру и рейтингу.

```text
Что лучше посмотреть вечером: комедию или триллер? Дай варианты с рейтингом выше 7.5
```

Ожидаемо: агент делает несколько поисков/фильтраций и дает рекомендацию.

## Архитектура

Ключевая часть находится в `src/kinomaniac/agent.py`:

```python
llm_with_tools = llm.bind_tools(tools)
chain = prompt | llm_with_tools
```

Системный prompt строгий: он описывает роль агента, правила проверки фактов, forced tool usage для сравнений, списков, рейтингов и запросов по конкретным фильмам.
Для Ollama `tool_choice` не используется как надежный механизм принуждения, поэтому в коде есть строгая policy-подсказка и retry guard для запросов, где инструмент обязателен.

Инструменты описаны в `src/kinomaniac/tools.py`, а низкоуровневый OMDb клиент в `src/kinomaniac/omdb.py`.
