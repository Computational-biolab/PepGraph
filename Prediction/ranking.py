import pandas as pd


def rank_predictions(results):

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Probability",
        ascending=False
    )

    df.insert(
        0,
        "Rank",
        range(1, len(df) + 1)
    )

    return df
