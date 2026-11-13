"""Baseline models and the label rules."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def majority_top3(y_train, n_rows):
    """Dumb baseline: predict the 3 most frequent labels for every row."""
    top = list(y_train.value_counts().index[:3])
    return [top] * n_rows


def train_baseline(texts, y, C=5.0):
    """TF-IDF (word + character n-grams) followed by multinomial logistic regression."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=100_000)),
    ])
    model = Pipeline([
        ("tfidf", features),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])
    model.fit(texts, y)
    return model


def allowed_mask(df, classes, labels_per_question=None):
    """Boolean matrix (rows x classes): which labels are possible for each row.

    - The True_/False_ prefix must match is_correct (whether the chosen answer is right).
    - If labels_per_question is given, the label must have been seen for that question in train.
    """
    label_is_true = np.array([c.startswith("True") for c in classes])
    is_correct = df["is_correct"].to_numpy().astype(bool)
    mask = label_is_true[None, :] == is_correct[:, None]
    if labels_per_question is not None:
        seen = np.array([
            [c in labels_per_question.get(q, set()) for c in classes]
            for q in df["QuestionId"]
        ])
        mask &= seen
    return mask


def top3_from_proba(proba, classes, allowed=None):
    """Turn class probabilities into the 3 best labels per row.

    Labels outside `allowed` are pushed to the end of the ranking.
    """
    proba = proba.copy()
    if allowed is not None:
        proba[~allowed] -= 1.0  # probabilities are in [0, 1], so these always rank last
    best = np.argsort(-proba, axis=1)[:, :3]
    return [list(classes[row]) for row in best]
