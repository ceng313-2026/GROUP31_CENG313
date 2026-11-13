"""Plotting, the competition metric (MAP@3) and submission helpers."""
import matplotlib.pyplot as plt
import pandas as pd


def bar_counts(series, title, ax=None, horizontal=False):
    """Bar chart of value counts, largest first."""
    counts = series.value_counts()
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    if horizontal:
        counts[::-1].plot.barh(ax=ax, color="#4C72B0")
        ax.set_xlabel("rows")
    else:
        counts.plot.bar(ax=ax, color="#4C72B0")
        ax.set_ylabel("rows")
    ax.set_title(title)
    return ax


def map_at_3(y_true, top3):
    """Mean Average Precision @ 3 with one correct label per row.

    A hit at rank 1, 2 or 3 scores 1, 1/2 or 1/3; a miss scores 0.
    """
    total = 0.0
    for truth, preds in zip(y_true, top3):
        for rank, pred in enumerate(preds[:3], start=1):
            if pred == truth:
                total += 1.0 / rank
                break
    return total / len(y_true)


def save_submission(row_ids, top3, path="submission.csv"):
    """Write the Kaggle submission: up to 3 space-separated labels per row."""
    sub = pd.DataFrame({
        "row_id": row_ids,
        "Category:Misconception": [" ".join(p) for p in top3],
    })
    sub.to_csv(path, index=False)
    return sub
