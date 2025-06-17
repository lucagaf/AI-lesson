import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "Data", "Test", "results.json")


def load_results(path: str) -> List[Dict[str, float]]:
    """Load leaderboard results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sort_leaderboard(results: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return leaderboard sorted by accuracy and mean confidence."""
    return sorted(results, key=lambda x: (x["accuracy"], x.get("mean_confidence", 0)), reverse=True)


def plot_leaderboard_anonymous(results: List[Dict[str, float]]) -> None:
    """Plot accuracy of all students without names, sorted descending."""
    lb = _sort_leaderboard(results)
    accuracies = [entry["accuracy"] for entry in lb]
    numbers = list(range(1, len(lb) + 1))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(numbers, accuracies, color="skyblue")

    plt.xlabel("Student")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy aller Teilnehmer:innen")
    plt.xticks(numbers, [str(n) for n in numbers])
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc * 100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("leaderboard_anonymous.png")
    plt.close()


def plot_podest_third_named(results: List[Dict[str, float]]) -> None:
    """Plot top3 with only third place named and bronze color."""
    lb = _sort_leaderboard(results)[:3]
    labels = ["1. Platz", "2. Platz", lb[2]["student"] if len(lb) >= 3 else ""]
    accuracies = [entry["accuracy"] for entry in lb]
    colors = ["skyblue", "skyblue", "#cd7f32"]  # bronze

    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(1, len(lb) + 1), accuracies, color=colors[:len(lb)])
    plt.xlabel("Platz")
    plt.ylabel("Accuracy (%)")
    plt.title("Podest")
    plt.xticks(range(1, len(lb) + 1), labels)
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc * 100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("podest_third_named.png")
    plt.close()


def plot_podest_second_third_named(results: List[Dict[str, float]]) -> None:
    """Plot top3 with second and third named; bronze and silver colors."""
    lb = _sort_leaderboard(results)[:3]
    labels = ["1. Platz", lb[1]["student"] if len(lb) >= 2 else "", lb[2]["student"] if len(lb) >= 3 else ""]
    accuracies = [entry["accuracy"] for entry in lb]
    colors = ["skyblue", "silver", "#cd7f32"]  # gold not requested

    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(1, len(lb) + 1), accuracies, color=colors[:len(lb)])
    plt.xlabel("Platz")
    plt.ylabel("Accuracy (%)")
    plt.title("Podest")
    plt.xticks(range(1, len(lb) + 1), labels)
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc * 100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("podest_second_third_named.png")
    plt.close()


def plot_podest_all_named(results: List[Dict[str, float]]) -> None:
    """Plot top3 with all names and gold, silver, bronze colors."""
    lb = _sort_leaderboard(results)[:3]
    labels = [entry["student"] for entry in lb]
    accuracies = [entry["accuracy"] for entry in lb]
    colors = ["gold", "silver", "#cd7f32"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(1, len(lb) + 1), accuracies, color=colors[:len(lb)])
    plt.xlabel("Platz")
    plt.ylabel("Accuracy (%)")
    plt.title("Podest")
    plt.xticks(range(1, len(lb) + 1), labels)
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc * 100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("podest_all_named.png")
    plt.close()


def main() -> None:
    results = load_results(RESULTS_PATH)
    plot_leaderboard_anonymous(results)
    plot_podest_third_named(results)
    plot_podest_second_third_named(results)
    plot_podest_all_named(results)


if __name__ == "__main__":
    main()
