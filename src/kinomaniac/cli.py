"""Command-line entry point for the Kinomaniac agent."""

from __future__ import annotations

import argparse
import json

from kinomaniac.agent import create_movie_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Киноманьяк: локальный movie-agent на Ollama + LangChain.")
    parser.add_argument("--trace", action="store_true", help="Print tool calls after each answer.")
    parser.add_argument("question", nargs="*", help="Movie question. If omitted, starts an interactive chat.")
    args = parser.parse_args()

    agent = create_movie_agent()

    if args.question:
        print(agent.ask(" ".join(args.question)))
        if args.trace:
            print_tool_trace(agent.last_tool_calls)
        return

    print("Киноманьяк готов. Напишите вопрос о кино или 'exit'.")
    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue
        print(f"\nКиноманьяк: {agent.ask(user_input)}")
        if args.trace:
            print_tool_trace(agent.last_tool_calls)


def print_tool_trace(tool_calls: list[dict]) -> None:
    print("\nTool calls:")
    if not tool_calls:
        print("[]")
        return
    print(json.dumps(tool_calls, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
