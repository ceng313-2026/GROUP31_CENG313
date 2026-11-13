"""Loading and preprocessing for the MAP - Charting Student Math Misunderstandings data."""
import re
from pathlib import Path

import pandas as pd


def load_data(data_dir="data"):
    """Read train.csv and test.csv from data_dir."""
    data_dir = Path(data_dir)
    # keep_default_na=False: the string "NA" in Misconception is a real label, not a missing value
    train = pd.read_csv(data_dir / "train.csv", keep_default_na=False, na_values=[""])
    test = pd.read_csv(data_dir / "test.csv", keep_default_na=False, na_values=[""])
    return train, test


def add_target(df):
    """Competition target = Category and Misconception joined with a colon."""
    df = df.copy()
    df["target"] = df["Category"] + ":" + df["Misconception"]
    return df


def find_correct_answers(train):
    """For every question, the MC answer that students labelled True_* chose.

    The True/False part of Category only says whether the chosen answer is right,
    so each question should have exactly one such answer.
    """
    is_true = train["Category"].str.startswith("True")
    return (
        train[is_true]
        .groupby("QuestionId")["MC_Answer"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )


def clean_text(text):
    """Lowercase, drop LaTeX delimiters and squeeze whitespace."""
    text = str(text).lower()
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df, correct_answers):
    """Add the features the baseline uses.

    is_correct: 1 if the student picked the right MC answer (known from train).
    text:       answer + correctness + explanation in one string for TF-IDF.
    """
    df = df.copy()
    df["StudentExplanation"] = df["StudentExplanation"].fillna("")
    df["is_correct"] = (
        df["MC_Answer"] == df["QuestionId"].map(correct_answers)
    ).astype(int)
    df["text"] = (
        "answer " + df["MC_Answer"].map(clean_text)
        + " correct " + df["is_correct"].map({1: "yes", 0: "no"})
        + " explanation " + df["StudentExplanation"].map(clean_text)
    )
    return df
