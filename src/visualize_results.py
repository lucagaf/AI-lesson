import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt


RESULTS_PATH = os.path.join(os.path.dirname(__file__), "Data", "Test", "results.json")


def load_results(path: str) -> List[Dict[str, float]]:
    """Load leaderboard results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_leaderboard_all(results: List[Dict[str, float]]) -> None:
    """Plot accuracy for all students including their names and values."""
    # results are assumed to be sorted already
    names = [entry["student"] for entry in results]
    accuracies = [entry["accuracy"] for entry in results]
    x = range(len(names))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, accuracies, color="skyblue")

    plt.xlabel("Student")
    plt.ylabel("Accuracy")

    # plt.title("Accuracy of all students")
    plt.xticks(x, names, rotation=45, ha="right")

    plt.title("Accuracy von allen Schüler:innen")
    # plt.xticks(x, [str(i) for i in x])

    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("leaderboard_all.png")
    plt.close()


def _sort_leaderboard(results: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return leaderboard sorted by accuracy and mean confidence."""
    return sorted(results, key=lambda x: (x["accuracy"], x.get("mean_confidence", 0)), reverse=True)


def plot_second_and_third(results: List[Dict[str, float]]) -> None:
    """Plot the accuracy of the 2nd and 3rd ranked students."""
    lb = _sort_leaderboard(results)
    top_entries = lb[1:3] if len(lb) >= 3 else lb[1:]

    names = [entry["student"] for entry in top_entries]
    accuracies = [entry["accuracy"] for entry in top_entries]

    x = range(len(top_entries))

    plt.figure(figsize=(6, 4))
    plt.bar(x, accuracies, color="steelblue")
    plt.xlabel("Student")

    plt.ylabel("Value")
    plt.title("Top 3 Schüler:innen")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("leaderboard_second_third.png")
    plt.close()


def plot_third_place(results: List[Dict[str, float]]) -> None:
    """Plot the accuracy of the 3rd ranked student."""
    lb = _sort_leaderboard(results)
    third = lb[2:3]

    if not third:
        return

    name = third[0]["student"]
    acc = third[0]["accuracy"]

    plt.figure(figsize=(4, 4))
    plt.bar([0], [acc], color="steelblue")
    plt.xlabel("Student")
    plt.ylabel("Accuracy")
    plt.title("3rd Place")
    plt.xticks([0], [name])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("leaderboard_third.png")
    plt.close()


def main() -> None:
    results = load_results(RESULTS_PATH)
    plot_leaderboard_all(results)
    plot_second_and_third(results)
    plot_third_place(results)


if __name__ == "__main__":
    main()
