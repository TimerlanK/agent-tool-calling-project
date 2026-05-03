"""Command-line entry point for the Kinomaniac agent."""

from __future__ import annotations

import argparse
import json
import time

from kinomaniac.agent import create_movie_agent, print_dialog


SEPARATOR_WIDTH = 72


def main() -> None:
    parser = argparse.ArgumentParser(description="Киноманьяк: movie-agent на OpenAI API + LangChain.")
    parser.add_argument("--trace", action="store_true", help="Печатать tool calls после каждого ответа.")
    parser.add_argument("--memory", action="store_true", help="Печатать summary buffer memory после каждого ответа.")
    parser.add_argument("question", nargs="*", help="Вопрос о кино. Если не указан, запускается интерактивный чат.")
    args = parser.parse_args()

    agent = create_movie_agent()

    if args.question:
        answer, elapsed_seconds = ask_with_timing(agent, " ".join(args.question))
        print(answer)
        print_execution_time(elapsed_seconds)
        if args.trace:
            print_tool_trace(agent.last_tool_calls)
        if args.memory:
            print_dialog(agent.summary, agent.recent_messages)
        print_turn_separator(1)
        return

    print("Киноманьяк готов. Напишите вопрос о кино или 'exit'.")
    turn_number = 1
    while True:
        user_input = input(f"\nВопрос {turn_number} - Вы: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue
        answer, elapsed_seconds = ask_with_timing(agent, user_input)
        print(f"\nКиноманьяк: {answer}")
        print_execution_time(elapsed_seconds)
        if args.trace:
            print_tool_trace(agent.last_tool_calls)
        if args.memory:
            print_dialog(agent.summary, agent.recent_messages)
        print_turn_separator(turn_number)
        turn_number += 1


def ask_with_timing(agent, question: str) -> tuple[str, float]:
    start_time = time.perf_counter()
    answer = agent.ask(question)
    return answer, time.perf_counter() - start_time


def print_execution_time(elapsed_seconds: float) -> None:
    print(f"\nВремя выполнения: {elapsed_seconds:.2f} сек.")


def print_tool_trace(tool_calls: list[dict]) -> None:
    print("\nTool calls:")
    if not tool_calls:
        print("[]")
        return
    print(json.dumps(tool_calls, ensure_ascii=False, indent=2))


def print_turn_separator(turn_number: int) -> None:
    label = f" КОНЕЦ ВОПРОСА {turn_number} "
    side_width = max((SEPARATOR_WIDTH - len(label)) // 2, 1)
    line = f"{'=' * side_width}{label}{'=' * side_width}"
    print(f"\n{line}\n")


if __name__ == "__main__":
    main()
