# GROUP31_CENG313

CENG313 Introduction to Data Science, course project

## Group

**Group ID:** 31

| Name | Student ID |
|---|---|
| Enes Tekneci | 23118080705 |
| Marziye Can | 23118080706 |

## Competition

[MAP – Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) (Kaggle, final submissions closed 15 Oct 2025, late submissions accepted)

From a student's multiple-choice answer and written explanation, predict `Category:Misconception`. For example, `True_Correct:NA` means a right answer with correct reasoning, and `False_Misconception:Incomplete` means a wrong answer whose explanation shows the `Incomplete` misconception. Up to 3 ranked guesses are allowed per row, scored with **MAP@3**.

## Scores

| Model | Validation MAP@3 (20 % hold-out) | Kaggle public | Kaggle private |
|---|---|---|---|
| Majority (3 most frequent labels) | 0.5403 | – | – |
| TF-IDF + Logistic Regression | 0.9167 | – | – |
| TF-IDF + Logistic Regression + EDA rules (Assignment 1 baseline) | 0.9196 | 0.90732 | 0.90760 |

Winning private score on the leaderboard: 0.94919. The late-submission screenshot goes in [`submissions/`](submissions/).

## Repository layout

```
├── assignment1_eda_baseline.ipynb   Assignment 1: EDA & baseline
├── assignment2_solution.ipynb       Assignment 2 (to be added)
├── kaggle_submit.py                 Self-contained script for the Kaggle late submission
├── src/
│   ├── data.py                      loading, target, is_correct feature, text cleaning
│   ├── models.py                    baselines, label rules
│   └── utils.py                     plots, MAP@3 metric, writing submission.csv
├── data/                            Kaggle data (not committed)
└── submissions/                     score screenshots
```

## Getting the data

1. Join the competition on Kaggle and accept the rules.
2. Download it from the competition's *Data* tab (**Download All**). With the Kaggle API instead, run `pip install kaggle` and put your API token (Kaggle → Settings → *Create New Token*) in `~/.kaggle/kaggle.json` first:
   ```bash
   kaggle competitions download -c map-charting-student-math-misunderstandings
   ```
3. Unzip into `data/` from the repository root, so that you have `data/train.csv`, `data/test.csv` and `data/sample_submission.csv`:
   ```bash
   unzip map-charting-student-math-misunderstandings.zip -d data
   ```
   On Windows, right-click the zip → *Extract All…* and choose the `data` folder.

## Running the notebooks

Python 3.12, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook assignment1_eda_baseline.ipynb
```

If GitHub does not render the notebook, open it on [nbviewer](https://nbviewer.org/github/enesTekneci/GROUP31_CENG313/blob/main/assignment1_eda_baseline.ipynb).

Run the notebook from the repository root so that `from src... import` works. A full *Restart & Run All* takes about 3–4 minutes on a laptop CPU, because the logistic regression is trained twice.

## Kaggle late submission

This is a code competition, so predictions must come from a Kaggle notebook (with internet turned off) that writes `submission.csv`. Paste [`kaggle_submit.py`](kaggle_submit.py) into a new notebook on the competition page, choose *Save Version → Save & Run All*, and submit the `submission.csv` from the version's output.
