# Kinomaniac: AI Movie Agent

Kinomaniac is a local AI movie expert built with **Ollama + LangChain LCEL + bind_tools**.
The agent verifies movie facts through the OMDb API.

## Features

- Search one movie by title.
- Search a list of movies.
- Compare two movies.
- Filter movies by genre.
- Find movies above a minimum IMDb rating.
- Multi-tool calling: the agent can call 2, 3, or more tools for one user request.
- Tools are created with `@tool`, use `requests`, handle errors with `try-except`, and return readable strings.
- Manual summary buffer memory: old conversation is compressed into a summary while recent messages stay fully available.

## Setup

1. Install Ollama: <https://ollama.com/>

2. Pull a local model with tool/function calling support:

```bash
ollama pull qwen2.5:7b-instruct
```

3. Create a virtual environment and install dependencies:

On Ubuntu/Debian you may need:

```bash
sudo apt install python3-venv python3-pip
```

Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Get an OMDb API key from <http://www.omdbapi.com/apikey.aspx>.

5. Create `.env`:

```bash
cp .env.example .env
```

Fill it:

```bash
OMDB_API_KEY=your_omdb_api_key
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
MEMORY_MAX_WORD_LIMIT=120
MEMORY_KEEP_LAST_MESSAGES=4
```

## Run

Interactive mode:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli
```

One question:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli "Compare the IMDb ratings of Matrix and Inception."
```

Show called tools:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --trace "Compare the IMDb ratings of Matrix and Inception."
```

Show current memory:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --memory
```

Check tools and memory together:

```bash
PYTHONPATH=src python3 -m kinomaniac.cli --trace --memory
```

First send a memory-only message:

```text
My name is Timur. I love science fiction and Christopher Nolan movies.
```

Then ask a multi-tool question:

```text
Compare the IMDb ratings of Matrix and Inception and recommend which one fits my taste better.
```

## Multi-Tool Examples

```text
Compare the IMDb ratings of Matrix and Inception.
```

Expected behavior: the agent calls the comparison tool or fetches details for both movies, then compares them.

```text
Find Batman movies, keep only thrillers, and show the best ones by IMDb rating.
```

Expected behavior: the agent searches a list, fetches details, filters by genre, and compares ratings.

```text
What is better for tonight: comedy or thriller? Give options with IMDb rating above 7.5.
```

Expected behavior: the agent performs multiple searches/filtering steps and gives a recommendation.

## Architecture

The key LCEL part is in `src/kinomaniac/agent.py`:

```python
intent_chain = intent_prompt | llm
llm_with_tools = llm.bind_tools(tools)
chain = chat_prompt | llm_with_tools
summary_chain = summary_prompt | llm
```

The agent first uses `intent_chain` as a small LLM router. It returns exactly one label:

- `MEMORY_ONLY`: remember the input without tool calls.
- `TOOL_NEEDED`: call OMDb tools before answering.
- `GENERAL_CHAT`: answer without tools for general movie discussion.

The system prompt is strict. It defines the agent role, fact-checking rules, forced tool usage for comparisons/lists/ratings/specific movie questions, and a no-tool path for memory-only user statements. The decision is made by the LLM router, not by hardcoded keyword matching.

For Ollama, `tool_choice` is not used as a reliable forcing mechanism. Instead, the code uses a strict tool policy hint and a retry guard when a tool is required.

Memory is implemented manually with a summary buffer pattern:

- `summary`: short compressed text for older conversation.
- `recent_messages`: the latest messages stored fully as `HumanMessage` and `AIMessage`.
- `chat_prompt`: system message with `{summary}`, `MessagesPlaceholder("recent_messages")`, and human message with `{input}`.
- `summary_prompt`: receives the old summary and `messages_to_summarize`, then returns an updated short summary.
- The summary preserves user name, preferences, favorite movies/genres, important facts, recent requests, and other useful context.
- `MEMORY_MAX_WORD_LIMIT` controls when compression happens.
- `MEMORY_KEEP_LAST_MESSAGES` controls how many latest messages stay fully available.

Tools are in `src/kinomaniac/tools.py`.
The low-level OMDb client is in `src/kinomaniac/omdb.py`.
