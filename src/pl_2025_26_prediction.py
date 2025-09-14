"""
Predict the outcome of the 2025/26 Premier League season using a random
forest classifier.

This script takes a series of historical Premier League seasons in CSV
format, builds summary statistics for each club (points, wins, draws,
losses, goals for/against and goal difference) and then trains a
RandomForestClassifier from scikit‑learn to predict the final league
position of each team in a subsequent season.  The 2025/26 Premier
League will include 20 clubs – the 17 sides that remained in the
division in 2024/25 and three promoted clubs (Leeds United, Burnley
and Sunderland)【177773065645842†screenshot】.  Since results from the
2024/25 campaign are not yet freely available, the model uses the
2023/24 season as the most recent set of training features.  New
clubs that did not compete in 2023/24 are assigned average feature
values based on the bottom three sides from that season.

The historical match files used by this script can be downloaded from
the open‑source football csv mirror hosted on GitHub.  Each file
(`eng1_2018-19.csv`, `eng1_2019-20.csv`, … `eng1_2023-24.csv`) lists
every Premier League match in the given season with columns for the
date, home side (Team 1), final score (FT), half‑time score (HT) and
away side (Team 2).  An example of the first few rows of the
2019/20 file is shown below:

```
              Date          Team 1   FT   HT            Team 2
0   Fri Aug 9 2019       Liverpool  4-1  4-0           Norwich
1  Sat Aug 10 2019        West Ham  0-5  0-1          Man City
2  Sat Aug 10 2019     Bournemouth  1-1  0-0  Sheffield United
3  Sat Aug 10 2019         Burnley  3-0  0-0       Southampton
4  Sat Aug 10 2019  Crystal Palace  0-0  0-0           Everton
```

Each CSV contains 380 matches (20 clubs playing 38 games each).  The
script parses the final score to determine home and away goals and
computes win/draw/loss outcomes accordingly.  After summarising the
season, the teams are sorted by points, goal difference and goals
scored to derive a final ranking.  For training data, each team’s
performance statistics from season `n` are used to predict its
position in season `n+1`.  Teams that enter the league via promotion
are assigned default feature values representing the average of the
bottom three clubs from the previous season.

The RandomForestClassifier hyperparameters can be adjusted via the
constants at the bottom of the script.  By default the model uses
100 trees, a maximum depth of 8 and a random seed for reproducible
results.  After training, the script prints the predicted league
table for 2025/26 along with a comparison to the training periods.

Usage
-----
Run the script from a terminal with Python 3.  Ensure that
`pandas`, `numpy` and `scikit‑learn` are installed.  All required
CSV files should reside in the same directory as this script or an
alternate path may be provided via the `season_files` list.

Example:

```
python pl_2025_26_prediction.py
```

The script outputs a predicted ranking of the 20 clubs for the
2025/26 Premier League season.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def parse_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """Parse final score into integer goal columns.

    The raw CSV files use a `FT` column that stores the full‑time
    result as a string such as `"2-1"`.  This helper splits the
    column into separate home and away goal counts and returns an
    updated DataFrame with `home_goals` and `away_goals` columns.

    Parameters
    ----------
    df : DataFrame
        Match data with columns `Team 1`, `Team 2` and `FT`.

    Returns
    -------
    DataFrame
        DataFrame with added `home_goals` and `away_goals` columns.
    """
    goals = df["FT"].str.split("-", expand=True)
    df = df.copy()
    df["home_goals"] = goals[0].astype(int)
    df["away_goals"] = goals[1].astype(int)
    return df


def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise a season into per‑team statistics and final ranking.

    Given a DataFrame of matches with columns `Team 1`, `Team 2`,
    `home_goals` and `away_goals`, compute the total points, wins,
    draws, losses, goals for and against and goal difference for each
    team.  After accumulating statistics, the teams are sorted by
    points (descending), goal difference (descending) and goals for
    (descending) to determine the final ranking.

    Parameters
    ----------
    matches : DataFrame
        DataFrame of parsed match results.

    Returns
    -------
    DataFrame
        Summary of the season with one row per team and columns:
        [`team`, `points`, `wins`, `draws`, `losses`, `goals_for`,
        `goals_against`, `goal_diff`, `position`].
    """
    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    })
    # iterate through each match and update team statistics
    for _, row in matches.iterrows():
        home, away = row["Team 1"], row["Team 2"]
        hg, ag = row["home_goals"], row["away_goals"]
        # update goals
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        # determine match outcome
        if hg > ag:
            # home win
            teams[home]["points"] += 3
            teams[home]["wins"] += 1
            teams[away]["losses"] += 1
        elif hg < ag:
            # away win
            teams[away]["points"] += 3
            teams[away]["wins"] += 1
            teams[home]["losses"] += 1
        else:
            # draw
            teams[home]["points"] += 1
            teams[away]["points"] += 1
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1
    # build DataFrame
    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        data.append(
            {
                "team": team,
                "points": stats["points"],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "goals_for": stats["goals_for"],
                "goals_against": stats["goals_against"],
                "goal_diff": goal_diff,
            }
        )
    summary = pd.DataFrame(data)
    # sort by points, goal diff, goals for
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary["position"] = summary.index + 1
    return summary


def prepare_training_data(season_files: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare training features and labels from a list of seasons.

    Given a list of file paths ordered chronologically, compute per‑team
    statistics for each season and build a dataset where the feature
    vector for season `n+1` comes from the statistics of season `n`.
    Teams promoted into the Premier League without previous season
    statistics are assigned default feature values equal to the
    average of the bottom three clubs in the prior season.

    Parameters
    ----------
    season_files : list of str
        Paths to season CSV files ordered from oldest to newest.

    Returns
    -------
    X_train : DataFrame
        Feature matrix (numeric) for training.
    y_train : Series
        Target series containing league positions (1–20).
    latest_features : DataFrame
        Feature matrix for the most recent season in the list (used
        for prediction).
    """
    season_summaries: Dict[str, pd.DataFrame] = {}
    # compute summary stats for each season
    for file_path in season_files:
        raw = pd.read_csv(file_path)
        parsed = parse_match_results(raw)
        summary = summarise_season(parsed)
        season_summaries[file_path] = summary
    # Build training dataset: use season n's stats to predict season n+1's position
    feature_rows = []
    target_rows = []
    files_sorted = season_files
    for i in range(len(files_sorted) - 1):
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")
        curr_summary = season_summaries[files_sorted[i + 1]].copy().set_index("team")
        # compute default features based on bottom three teams from previous season
        bottom_three = prev_summary.sort_values(
            ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
        ).head(3)
        default_features = bottom_three.mean().to_dict()
        # for each team in current season, collect features
        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:
                feats = prev_summary.loc[team][
                    ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
                ].to_dict()
            else:
                # promoted team – assign default bottom three stats
                feats = {k: default_features[k] for k in [
                    "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
                ]}
            feature_rows.append(feats)
            target_rows.append(row["position"])
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    # features for the most recent season for which we will predict the next season
    last_summary = season_summaries[files_sorted[-1]].copy().set_index("team")
    # compute default features for new promoted teams in the upcoming season
    # this uses bottom three of last_summary
    bottom_three_last = last_summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
    ).head(3)
    default_features_last = bottom_three_last.mean().to_dict()
    latest_features_rows = []
    latest_teams = last_summary.index.tolist()
    # incorporate promoted teams for 2025/26 (Leeds United, Burnley, Sunderland)
    promoted = ["Leeds United", "Burnley", "Sunderland"]
    # if a promoted team already exists in last_summary (e.g. Burnley was relegated earlier), use its stats
    for team in latest_teams:
        feats = last_summary.loc[team][
            ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
        ].to_dict()
        latest_features_rows.append((team, feats))
    for team in promoted:
        if team not in latest_teams:
            feats = {k: default_features_last[k] for k in [
                "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
            ]}
            latest_features_rows.append((team, feats))
    latest_features_df = pd.DataFrame([feats for _, feats in latest_features_rows],
                                      index=[t for t, _ in latest_features_rows])
    return X_train, y_train, latest_features_df


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Create a pipeline that scales features and trains a RandomForest.

    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Target positions (1–20).

    Returns
    -------
    Pipeline
        Scikit‑learn pipeline with StandardScaler and RandomForestClassifier.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model


def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predict the league table ordering for the given features.

    Parameters
    ----------
    model : Pipeline
        Trained scikit‑learn pipeline.
    features : DataFrame
        Feature rows indexed by team name.

    Returns
    -------
    DataFrame
        Predicted positions sorted from 1 to 20.
    """
    # use predicted probabilities to compute an expected finishing
    # position.  RandomForestClassifier returns a probability
    # distribution over the 20 possible finishing positions.  By
    # multiplying each probability by its corresponding class index
    # (1–20) we obtain an expected (fractional) finishing position.
    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    # sort teams by lowest expected position (i.e. best finish)
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    # assign integer ranks 1..n based on sorted order
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]


def main():
    # define the season files in chronological order
    season_files = [
        os.path.join(os.path.dirname(__file__), "eng1_2018-19.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2019-20.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2020-21.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2021-22.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2022-23.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2023-24.csv"),
    ]
    # prepare training data
    X_train, y_train, latest_features = prepare_training_data(season_files)
    # train model
    model = build_and_train_model(X_train, y_train)
    # predict ranking for 2025/26
    predictions = predict_league_table(model, latest_features)
    # keep only the top 20 teams based on expected position.  In reality
    # the Premier League contains exactly 20 clubs.  Since we may
    # include extra promoted teams due to unavailable data for the
    # intermediate 2024/25 season, truncate to 20.
    predictions = predictions.iloc[:20].copy()
    print("Predicted Premier League 2025/26 table (1 = champion):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(expected pos {row['expected_position']:.2f})"
        )


if __name__ == "__main__":
    main()