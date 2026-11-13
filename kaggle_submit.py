import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# Kaggle has used more than one input folder layout, so search for the file
paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert paths, "train.csv not found: attach the competition data under Input"
# another attached dataset may also have a train.csv
train_path = next((p for p in paths if "map-charting" in p), paths[0])
data_dir = os.path.dirname(train_path)
train = pd.read_csv(f"{data_dir}/train.csv", keep_default_na=False, na_values=[""])
test = pd.read_csv(f"{data_dir}/test.csv", keep_default_na=False, na_values=[""])

# preprocessing, as in notebook section 4 and src/data.py
train["target"] = train["Category"] + ":" + train["Misconception"]
train = train.drop_duplicates(
    subset=["QuestionId", "MC_Answer", "StudentExplanation", "target"]).reset_index(drop=True)

is_true = train["Category"].str.startswith("True")
correct_answers = (train[is_true].groupby("QuestionId")["MC_Answer"]
                   .agg(lambda s: s.value_counts().index[0]).to_dict())


def clean_text(text):
    text = str(text).lower().replace("\\(", " ").replace("\\)", " ")
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df):
    df = df.copy()
    df["StudentExplanation"] = df["StudentExplanation"].fillna("")
    df["is_correct"] = (df["MC_Answer"] == df["QuestionId"].map(correct_answers)).astype(int)
    df["text"] = ("answer " + df["MC_Answer"].map(clean_text)
                  + " correct " + df["is_correct"].map({1: "yes", 0: "no"})
                  + " explanation " + df["StudentExplanation"].map(clean_text))
    return df


train, test = preprocess(train), preprocess(test)

# model, as in src/models.py
model = Pipeline([
    ("tfidf", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=100_000)),
    ])),
    ("clf", LogisticRegression(C=5.0, max_iter=2000)),
])
model.fit(train["text"], train["target"])
classes = model.classes_
proba = model.predict_proba(test["text"])

# EDA rules: True/False must match is_correct, and the label must have been seen for the question
labels_per_question = train.groupby("QuestionId")["target"].agg(set).to_dict()
label_is_true = np.array([c.startswith("True") for c in classes])
allowed = label_is_true[None, :] == test["is_correct"].to_numpy().astype(bool)[:, None]
allowed &= np.array([[c in labels_per_question.get(q, set()) for c in classes]
                     for q in test["QuestionId"]])
proba[~allowed] -= 1.0

top3 = np.argsort(-proba, axis=1)[:, :3]
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": [" ".join(classes[row]) for row in top3],
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
