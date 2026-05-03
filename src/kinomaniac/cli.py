"""Command-line entry point for the Kinomaniac agent."""

from __future__ import annotations

import argparse
import json

from kinomaniac.agent import create_movie_agent, print_dialog


SEPARATOR_WIDTH = 72


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinomaniac: local movie agent on Ollama + LangChain.")
    parser.add_argument("--trace", action="store_true", help="Print tool calls after each answer.")
    parser.add_argument("--memory", action="store_true", help="Print summary buffer memory after each answer.")
    parser.add_argument("question", nargs="*", help="Movie question. If omitted, starts an interactive chat.")
    args = parser.parse_args()

    agent = create_movie_agent()

    if args.question:
        print(agent.ask(" ".join(args.question)))
        if args.trace:
            print_tool_trace(agent.last_tool_calls)
        if args.memory:
            print_dialog(agent.summary, agent.recent_messages)
        print_turn_separator(1)
        return

    print("Kinomaniac is ready. Ask a movie question or type 'exit'.")
    turn_number = 1
    while True:
        user_input = input(f"\nTurn {turn_number} - You: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue
        print(f"\nKinomaniac: {agent.ask(user_input)}")
        if args.trace:
            print_tool_trace(agent.last_tool_calls)
        if args.memory:
            print_dialog(agent.summary, agent.recent_messages)
        print_turn_separator(turn_number)
        turn_number += 1


def print_tool_trace(tool_calls: list[dict]) -> None:
    print("\nTool calls:")
    if not tool_calls:
        print("[]")
        return
    print(json.dumps(tool_calls, ensure_ascii=False, indent=2))


def print_turn_separator(turn_number: int) -> None:
    label = f" END OF TURN {turn_number} "
    side_width = max((SEPARATOR_WIDTH - len(label)) // 2, 1)
    line = f"{'=' * side_width}{label}{'=' * side_width}"
    print(f"\n{line}\n")


if __name__ == "__main__":
    main()
