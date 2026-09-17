"""Markdown table from games_eval json files: python -m decider.games_table label=path ..."""
import json, sys


def _cli():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    data = {k: json.load(open(p)) for k, p in runs}
    first = data[runs[0][0]]
    games = list(first)
    print("| game | split | random | teacher | " + " | ".join(k for k, _ in runs) + " |")
    print("|---|---|---|---|" + "---|" * len(runs))
    for g in games:
        row = first[g]
        print(f"| {g} | {'train' if row['train'] else 'held-out'} | {row['random']:.2f} | {row['teacher']:.2f} | " + " | ".join(f"{data[k][g]['model']:.2f}" if g in data[k] else "" for k, _ in runs) + " |")


if __name__ == "__main__":
    _cli()
