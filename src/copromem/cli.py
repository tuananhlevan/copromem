"""Command-line entry point."""
from __future__ import annotations

import argparse
from .pipeline import run
from .agent_run import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone CoProMem")
    parser.add_argument("command_or_config", help="'agent', 'offline', or an offline config path")
    parser.add_argument("config", nargs="?", help="run configuration JSON")
    args = parser.parse_args()
    if args.command_or_config == "agent":
        if not args.config:
            parser.error("agent requires a config path")
        report = run_agent(args.config)
        print(f"CoProMem: {len(report['results'])} agent tasks")
    else:
        if args.command_or_config == "offline":
            if not args.config:
                parser.error("offline requires a config path")
            path = args.config
        else:
            if args.config:
                parser.error("unexpected second argument")
            path = args.command_or_config
        report = run(path)
        print(f"CoProMem: {len(report['updates'])} batches, {len(report['results'])} queries")


if __name__ == "__main__":
    main()
