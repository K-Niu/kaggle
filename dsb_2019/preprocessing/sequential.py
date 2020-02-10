from typing import List, Dict, Any
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from dsb_2019.preprocessing.constants import (
    EVENT_CODES,
    CORRECT_EVENT_CODES,
    EVENT_DATA_DETAILS,
)


def extract_features(
    gameplay_data: pd.DataFrame, test: bool = False
) -> List[Dict[str, Any]]:
    type_sequence = []
    title_sequence = []
    world_sequence = []
    day_of_week_sequence = []
    hour_sequence = []
    time_since_first_game_session_sequence = []

    event_code_counts_sequences = {event_code: [] for event_code in EVENT_CODES}
    event_code_correct_sequences = {
        event_code: [] for event_code in CORRECT_EVENT_CODES
    }
    event_code_attempts_sequences = {
        event_code: [] for event_code in CORRECT_EVENT_CODES
    }

    all_assessments = []
    for i, session in gameplay_data.groupby("game_session", sort=False):
        installation_id = session["installation_id"].iloc[0]
        session_type = session["type"].iloc[0]
        session_title = session["title"].iloc[0]
        session_world = session["world"].iloc[0]
        session_day_of_week = session["timestamp"].iloc[0].strftime("%a")
        session_hour = session["timestamp"].iloc[0].hour
        time_since_first_game_session = (
            session.iloc[0]["timestamp"] - gameplay_data.iloc[0]["timestamp"]
        ).total_seconds()

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
                "types": type_sequence.copy(),
                "titles": title_sequence.copy(),
                "worlds": world_sequence.copy(),
                "days_of_week": day_of_week_sequence.copy(),
                "hours": hour_sequence.copy(),
                "times_since_first_game_session": time_since_first_game_session_sequence.copy(),
            }
            features.update(
                {
                    f"{event_code}_counts": event_code_counts_sequences[
                        event_code
                    ].copy()
                    for event_code in EVENT_CODES
                }
            )
            features.update(
                {
                    f"{event_code}_corrects": event_code_correct_sequences[
                        event_code
                    ].copy()
                    for event_code in CORRECT_EVENT_CODES
                }
            )
            features.update(
                {
                    f"{event_code}_attempts": event_code_attempts_sequences[
                        event_code
                    ].copy()
                    for event_code in CORRECT_EVENT_CODES
                }
            )

            if test:
                all_assessments.append(features)
            elif len(assessment_attempts) > 0:
                features.update({"accuracy_group": accuracy_group})
                all_assessments.append(features)
            # Maybe add an incomplete assessments feature here

        type_sequence.append(session_type)
        title_sequence.append(session_title)
        world_sequence.append(session_world)
        day_of_week_sequence.append(session_day_of_week)
        hour_sequence.append(session_hour)
        time_since_first_game_session_sequence.append(time_since_first_game_session)

        session_event_codes = session.groupby("event_code")["installation_id"].agg(
            "count"
        )
        for event_code in EVENT_CODES:
            if event_code in session_event_codes.index:
                event_code_counts_sequences[event_code].append(
                    session_event_codes[event_code]
                )
            else:
                event_code_counts_sequences[event_code].append(0)

        corrects = (
            session[~session["correct"].isna()]
            .groupby("event_code")["correct"]
            .agg("sum")
            .astype(np.int64)
        )
        attempts = (
            session[~session["correct"].isna()]
            .groupby("event_code")["correct"]
            .agg("count")
        )
        for event_code in CORRECT_EVENT_CODES:
            if event_code in corrects.index:
                event_code_correct_sequences[event_code].append(corrects[event_code])
            else:
                event_code_correct_sequences[event_code].append(0)

            if event_code in attempts.index:
                event_code_correct_sequences[event_code].append(attempts[event_code])
            else:
                event_code_attempts_sequences[event_code].append(0)

    if test:
        return [all_assessments[-1]]
    return all_assessments


if __name__ == "__main__":
    train = pd.read_csv("dsb_2019/data/train.csv")
    test = pd.read_csv("dsb_2019/data/test.csv")

    # Process train dataset
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

    # Process test dataset
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

    train_features_df.to_pickle("dsb_2019/data/sequential_train_features.pkl")
    test_features_df.to_pickle("dsb_2019/data/sequential_test_features.pkl")
