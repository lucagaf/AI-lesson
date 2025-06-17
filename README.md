# AI-lesson
This project is part of a hands-on lesson designed for 4th to 6th grade students to introduce them to the world of artificial intelligence. Using Google's Teachable Machine, students create their own image classification models by training them on pictures of five different animals.

## Visualising evaluation results
After running `Evaluate_Student_Submissions.py`, the evaluation metrics are stored in `src/Data/Test/results.json`. The script `src/visualize_results.py` reads this file and creates three charts:

1. **Leaderboard of all students** – shows the accuracy for every participant with their name and the value displayed on top of each bar.
2. **Second and third place** – displays only the names and accuracies of the second and third ranked students.
3. **Third place only** – shows the accuracy of the student ranked third.

Run the visualisation with:

```bash
python src/visualize_results.py
```

The generated images `leaderboard_all.png`, `leaderboard_second_third.png` and `leaderboard_third.png` will appear in the current directory.
