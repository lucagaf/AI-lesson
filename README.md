# AI-lesson
This project is part of a hands-on lesson designed for 4th to 6th grade students to introduce them to the world of artificial intelligence. Using Google's Teachable Machine, students create their own image classification models by training them on pictures of five different animals.

## Visualising evaluation results
After running `Evaluate_Student_Submissions.py`, the evaluation metrics are stored in `src/Data/Test/results.json`. The script `src/visualize_results.py` reads this file and creates two charts:

1. **Accuracy of all students** – displays the accuracy for each student anonymously.
2. **Top three students** – shows both accuracy and mean precision for the three best students including their names.

Run the visualisation with:

```bash
python src/visualize_results.py
```

The generated images `accuracy_overall.png` and `top3_accuracy_precision.png` will appear in the current directory.
