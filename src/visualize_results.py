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


def plot_overall_accuracy(results: List[Dict[str, float]]) -> None:
    """Plot accuracy for all students without showing their names."""
    accuracies = [entry["accuracy"] for entry in results]
    x = range(1, len(accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(x, accuracies, color="skyblue")
    plt.xlabel("Student")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of all students")
    plt.xticks(x, [str(i) for i in x])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("accuracy_overall.png")
    plt.close()


def plot_top3(results: List[Dict[str, float]]) -> None:
    """Plot accuracy and mean precision of the best three students."""
    # Sort by accuracy then mean_confidence
    sorted_results = sorted(results, key=lambda x: (x["accuracy"], x["mean_confidence"]), reverse=True)
    top3 = sorted_results[:3]

    names = [entry["student"] for entry in top3]
    accuracies = [entry["accuracy"] for entry in top3]
    precisions = [entry["mean_confidence"] for entry in top3]

    x = range(len(top3))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([xi - width/2 for xi in x], accuracies, width=width, label="Accuracy", color="steelblue")
    plt.bar([xi + width/2 for xi in x], precisions, width=width, label="Mean Precision", color="orange")

    plt.xlabel("Student")
    plt.ylabel("Value")
    plt.title("Top 3 Students")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("top3_accuracy_precision.png")
    plt.close()


def main() -> None:
    results = load_results(RESULTS_PATH)
    plot_overall_accuracy(results)
    plot_top3(results)


if __name__ == "__main__":
    main()
