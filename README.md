# Киноманьяк: AI-агент по фильмам

Киноманьяк — AI-агент эксперт по кино на **OpenAI API + LangChain LCEL + bind_tools**.
Факты о фильмах агент проверяет через OMDb API.

## Возможности

- Поиск одного фильма по названию.
- Поиск списка фильмов.
- Сравнение двух и более фильмов через отдельный вызов `search_movie_by_title` для каждого фильма.
- Получение точной карточки фильма по IMDb ID из результатов поиска.
- Получение детальных карточек пачкой через `get_movie_details_batch`.
- Проверка LLM-generated candidate titles через OMDb для запросов по актеру/персоне и жанровых рекомендаций, потому что OMDb не умеет искать напрямую по актеру или выдавать топ фильмов по жанру.
- Фильтрация по жанру через `filter_movies_by_genre`.
- Фильтрация по минимальному IMDb rating через `filter_movies_by_min_rating`.
- Сортировка по IMDb rating через `sort_movies_by_imdb_rating`.
- Multi-tool calling: агент может вызвать 2, 3 и больше tools для одного запроса.
- Tools созданы через `@tool`, используют `requests`, обрабатывают ошибки через `try-except` и возвращают читаемые строки.
- Ручная summary buffer memory: старый диалог сжимается в summary, а последние сообщения хранятся полностью.

## Установка

1. Создайте виртуальное окружение и установите зависимости:

На Ubuntu/Debian может понадобиться:

```bash
sudo apt install python3-venv python3-pip
```

Затем:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Получите OMDb API key: <http://www.omdbapi.com/apikey.aspx>.

3. Получите OpenAI API key: <https://platform.openai.com/api-keys>.

4. Создайте `.env`:

```bash
cp .env.example .env
```

Заполните:

```bash
OMDB_API_KEY=your_omdb_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.4-mini
OPENAI_TEMPERATURE=0.2
MAX_TOOL_ROUNDS=8
MEMORY_MAX_WORD_LIMIT=120
MEMORY_KEEP_LAST_MESSAGES=4
```

## Запуск

Интерактивный режим:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli
```

Один вопрос:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli "Сравни IMDb рейтинги Matrix и Inception."
```

Показать вызванные tools:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --trace "Сравни IMDb рейтинги Matrix и Inception."
```

Показать текущую память:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --memory
```

Проверить tools и память вместе:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --trace --memory
```

Сначала отправьте сообщение, которое агент должен запомнить:

```text
Меня зовут Тимур. Я люблю научную фантастику и фильмы Кристофера Нолана.
```

Потом задайте multi-tool вопрос:

```text
Сравни IMDb рейтинги Matrix и Inception и посоветуй, что мне больше подойдет.
```

## Примеры multi-tool запросов

```text
Сравни IMDb рейтинги Matrix и Inception.
```

Ожидаемо: агент вызывает `search_movie_by_title` для обоих фильмов, затем сравнивает их.

```text
Найди фильмы Batman, оставь только триллеры и покажи лучшие по IMDb рейтингу.
```

Ожидаемо: агент вызывает `search_movie_list`, затем `get_movie_details_batch`, затем `filter_movies_by_genre`, затем `sort_movies_by_imdb_rating`.

```text
Найди Batman фильмы с IMDb rating выше 8.
```

Ожидаемо: агент вызывает `search_movie_list`, затем `get_movie_details_batch`, затем `filter_movies_by_min_rating`, затем `sort_movies_by_imdb_rating`.

```text
Сравни Matrix, Inception и Interstellar по IMDb рейтингу и жанру.
```

Ожидаемо: агент вызывает `search_movie_by_title` один раз для каждого фильма, затем сравнивает все полученные карточки.

```text
Найди фильмы с Леонардо ДиКаприо и покажи их рейтинги.
```

Ожидаемо: агент не вызывает `search_movie_list` с именем актера, потому что OMDb не поддерживает actor search. Вместо этого LLM предлагает candidate titles из своих знаний, затем вызывает `verify_candidate_movie_titles`, который проверяет эти названия и поле `Actors` через OMDb. В ответе агент предупреждает, что candidate titles были сгенерированы моделью, а детали и рейтинги проверены через OMDb.

```text
Что лучше посмотреть вечером: комедию или триллер? Дай варианты с IMDb rating выше 7.5.
```

Ожидаемо: агент составляет отдельные candidate titles для комедий и триллеров из знаний LLM, затем вызывает `verify_candidate_movie_titles` отдельно для каждого жанра с `required_genres="Comedy"` и `required_genres="Thriller"`, проверяет рейтинги через OMDb и дает одну рекомендацию. Агент не должен искать `search_movie_list("Comedy")`, потому что OMDb ищет по названию, а не по топу жанра.

## Архитектура

Ключевая LCEL-часть находится в `src/kinomaniac/agent.py`:

```python
intent_chain = intent_prompt | llm
llm_with_tools = llm.bind_tools(tools)
chain = chat_prompt | llm_with_tools
summary_chain = summary_prompt | llm
```

Сначала агент использует `intent_chain` как маленький LLM router. Он возвращает ровно один label:

- `MEMORY_ONLY`: запомнить сообщение без tool calls.
- `TOOL_NEEDED`: вызвать OMDb tools перед ответом.
- `GENERAL_CHAT`: ответить без tools для общего разговора о кино.

System prompt строгий: он описывает роль агента, правила проверки фактов, forced tool usage для сравнений/списков/рейтингов/конкретных фильмов и no-tool путь для сообщений, которые нужно только запомнить. Решение принимает LLM router, а не hardcoded keyword matching.

Фильтрация и ранжирование намеренно сделаны через несколько видимых tool calls. Например, запрос про Batman-триллеры должен идти как pipeline:

```text
search_movie_list
-> get_movie_details_batch
-> filter_movies_by_genre
-> sort_movies_by_imdb_rating
```

А запрос с минимальным рейтингом:

```text
search_movie_list
-> get_movie_details_batch
-> filter_movies_by_min_rating
-> sort_movies_by_imdb_rating
```

Код использует строгую tool policy подсказку и retry guard, когда tool обязателен.

Память реализована вручную через summary buffer pattern:

- `summary`: короткое сжатие старого диалога.
- `recent_messages`: последние сообщения полностью как `HumanMessage` и `AIMessage`.
- `chat_prompt`: system message с `{summary}`, `MessagesPlaceholder("recent_messages")`, human message с `{input}`.
- `summary_prompt`: получает старую summary и `messages_to_summarize`, затем возвращает обновленную короткую summary.
- Summary сохраняет имя пользователя, предпочтения, любимые фильмы/жанры, важные факты, недавние запросы и другой полезный контекст.
- `MEMORY_MAX_WORD_LIMIT` управляет моментом сжатия.
- `MEMORY_KEEP_LAST_MESSAGES` управляет тем, сколько последних сообщений хранить полностью.

Tools находятся в `src/kinomaniac/tools.py`.
Низкоуровневый OMDb client находится в `src/kinomaniac/omdb.py`.
