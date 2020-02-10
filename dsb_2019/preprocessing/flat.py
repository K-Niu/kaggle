from typing import List, Dict, Any
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from utils.feature_selection import (
    remove_correlated_features,
    remove_high_entropy_features,
)
from dsb_2019.preprocessing.utils import clean_column_name
from dsb_2019.preprocessing.constants import (
    TITLES,
    EVENT_CODES,
    EVENT_TYPES,
    TITLE_EVENT_CODES,
    WORLDS,
    CLIP_TIMES,
    ROUND_EVENTS,
    LEVEL_EVENTS,
    CORRECT_EVENTS,
    EVENT_DATA_DETAILS,
)


def extract_features(
    gameplay_data: pd.DataFrame, test: bool = False
) -> List[Dict[str, Any]]:
    title_event_code_counts = {
        title_event_code: 0 for title_event_code in TITLE_EVENT_CODES
    }
    type_counts = {event_type: 0 for event_type in EVENT_TYPES}
    event_code_counts = {event_code: 0 for event_code in EVENT_CODES}
    title_counts = {title: 0 for title in TITLES}
    world_counts = {world: 0 for world in WORLDS}
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}

    max_event_rounds = {event: 0 for event in ROUND_EVENTS}
    max_event_levels = {event: 0 for event in LEVEL_EVENTS}
    event_corrects = {event: 0 for event in CORRECT_EVENTS}
    event_attempts = {event: 0 for event in CORRECT_EVENTS}

    session_count = 0
    event_count = 0

    type_session_durations = {event_type: [] for event_type in EVENT_TYPES}

    all_assessments = []
    for i, session in gameplay_data.groupby("game_session", sort=False):
        installation_id = session["installation_id"].iloc[0]
        session_type = session["type"].iloc[0]
        session_title = session["title"].iloc[0]
        session_world = session["world"].iloc[0]

        if session_type == "Assessment":
            if session_title == "Bird Measurer (Assessment)":
                assessment_attempts = session.query("event_code == 4110")
            else:
                assessment_attempts = session.query("event_code == 4100")

            true_attempts = len(
                assessment_attempts[assessment_attempts["correct"] == True]
            )
            false_attempts = len(
                assessment_attempts[assessment_attempts["correct"] == False]
            )
            accuracy = (
                true_attempts / (true_attempts + false_attempts)
                if true_attempts or false_attempts
                else 0
            )
            if accuracy == 0:
                accuracy_group = 0
            elif accuracy == 1:
                accuracy_group = 3
            elif accuracy == 0.5:
                accuracy_group = 2
            else:
                accuracy_group = 1
            features = {
                "installation_id": installation_id,
                "assessment": session_title,
                "session_count": session_count,
                "event_count": event_count,
                "accuracy_group_mean": (
                    accuracy_groups[1] + 2 * accuracy_groups[2] + 3 * accuracy_groups[3]
                )
                / (
                    accuracy_groups[0]
                    + accuracy_groups[1]
                    + accuracy_groups[2]
                    + accuracy_groups[3]
                )
                if (
                    accuracy_groups[0]
                    + accuracy_groups[1]
                    + accuracy_groups[2]
                    + accuracy_groups[3]
                )
                else None,
            }
            features.update(
                {
                    f"{title_event_code}_count": title_event_code_counts[
                        title_event_code
                    ]
                    for title_event_code in TITLE_EVENT_CODES
                }
            )
            features.update(
                {
                    f"{event_type}_count": type_counts[event_type]
                    for event_type in EVENT_TYPES
                }
            )
            features.update(
                {
                    f"{event_code}_count": event_code_counts[event_code]
                    for event_code in EVENT_CODES
                }
            )
            features.update({f"{title}_count": title_counts[title] for title in TITLES})
            features.update({f"{world}_count": world_counts[world] for world in WORLDS})
            features.update(
                {
                    f"accuracy_group_{group}": accuracy_groups[group]
                    for group in [0, 1, 2, 3]
                }
            )

            # Event detail based features
            features.update(
                {
                    f"{event}_max_round": max_event_rounds[event]
                    for event in ROUND_EVENTS
                }
            )
            features.update(
                {
                    f"{event}_max_level": max_event_levels[event]
                    for event in LEVEL_EVENTS
                }
            )
            features.update(
                {f"{event}_corrects": event_corrects[event] for event in CORRECT_EVENTS}
            )
            features.update(
                {f"{event}_attempt": event_attempts[event] for event in CORRECT_EVENTS}
            )

            # Be careful because 0 should be NA
            features.update(
                {
                    f"{event_type}_duration_mean": np.mean(
                        type_session_durations[event_type]
                    )
                    if type_session_durations[event_type]
                    else 0
                    for event_type in EVENT_TYPES
                }
            )
            # Be careful because 0 should be NA
            features.update(
                {
                    f"{event_type}_duration_std": np.std(
                        type_session_durations[event_type]
                    )
                    if type_session_durations[event_type]
                    else 0
                    for event_type in EVENT_TYPES
                }
            )

            if test:
                all_assessments.append(features)
            elif len(assessment_attempts) > 0:
                features.update({"accuracy_group": accuracy_group})
                all_assessments.append(features)
                accuracy_groups[accuracy_group] += 1
            # Maybe add an incomplete assessments feature here

        for _, event in session.iterrows():
            title_event_code = event["title_event_code"]
            title_event_code_counts[title_event_code] += 1
            if event["correct"] == True:
                event_corrects[title_event_code] += 1
                event_attempts[title_event_code] += 1
            if event["correct"] == False:
                event_attempts[title_event_code] += 1
            if not np.isnan(event["round"]):
                max_event_rounds[title_event_code] = max(
                    max_event_rounds[title_event_code], event["round"]
                )
            if not np.isnan(event["level"]):
                max_event_levels[title_event_code] = max(
                    max_event_levels[title_event_code], event["level"]
                )
            event_code_counts[event["event_code"]] += 1

        type_counts[session_type] += 1
        title_counts[session_title] += 1
        world_counts[session_world] += 1

        session_count += 1
        event_count += len(session)

        if session_type == "Clip":
            type_session_durations["Clip"].append(CLIP_TIMES[session_title])
        else:
            type_session_durations[session_type].append(
                (
                    session.iloc[-1]["timestamp"] - session.iloc[0]["timestamp"]
                ).total_seconds()
            )

    if test:
        return [all_assessments[-1]]
    return all_assessments


if __name__ == "__main__":
    train = pd.read_csv("dsb_2019/data/train.csv")
    test = pd.read_csv("dsb_2019/data/test.csv")

    # Process train dataset
    train["title_event_code"] = (
        train["title"].astype(str) + "_" + train["event_code"].astype(str)
    )
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    for detail in EVENT_DATA_DETAILS:
        train[detail] = train["event_data"].apply(lambda x: json.loads(x).get(detail))

    with Pool(processes=cpu_count()) as pool:
        train_features = []
        result = [
            pool.apply_async(extract_features, kwds={"gameplay_data": events})
            for _, events in train.groupby("installation_id")
        ]
        for i, res in enumerate(result):
            train_features += res.get()
            if i % 1000 == 0:
                print(f"Processed {i} installation IDs")

    train_features_df = pd.DataFrame(train_features)
    train_features_df = train_features_df.rename(
        columns={
            col: clean_column_name(col) for col in train_features_df.columns.values
        }
    )

    # Process test dataset
    test["title_event_code"] = (
        test["title"].astype(str) + "_" + test["event_code"].astype(str)
    )
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    for detail in EVENT_DATA_DETAILS:
        test[detail] = test["event_data"].apply(lambda x: json.loads(x).get(detail))

    with Pool(processes=cpu_count()) as pool:
        test_features = []
        result = [
            pool.apply_async(
                extract_features, kwds={"gameplay_data": events, "test": True}
            )
            for _, events in test.groupby("installation_id")
        ]
        for res in result:
            test_features += res.get()

    test_features_df = pd.DataFrame(test_features)
    test_features_df = test_features_df.rename(
        columns={col: clean_column_name(col) for col in test_features_df.columns.values}
    )

    # Feature selection
    correlated_features = remove_correlated_features(
        train_features_df,
        train_features_df.columns.difference(
            ["installation_id", "accuracy_group", "assessment"]
        ),
    )
    high_entropy_features = remove_high_entropy_features(
        train_features_df,
        train_features_df.columns.difference(
            ["installation_id", "accuracy_group", "assessment"]
        ),
    )

    train_features_df = train_features_df.drop(
        columns=correlated_features + high_entropy_features
    )
    test_features_df = test_features_df.drop(
        columns=correlated_features + high_entropy_features
    )

    train_features_df.to_csv("dsb_2019/data/flat_train_features.csv", index=False)
    test_features_df.to_csv("dsb_2019/data/flat_test_features.csv", index=False)
