import pandas as pd
import numpy as np
import uuid
import random

def generate_mock_df(datatypes, n_rows=500):
    rows = []

    for _ in range(n_rows):
        row = {}
        for _, r in datatypes.iterrows():
            col, dtype = r["column"], r["datatype"]

            if "_status" in col:
                row[col] = np.random.choice([True, False], p=[0.1, 0.9])
            elif dtype == "bool":
                row[col] = random.choice([True, False])
            elif "followup" in col:
                row[col] = np.random.uniform(0,12)
            elif dtype == "float64":
                row[col] = np.random.uniform(-5, 5)
            elif dtype == "int64":
                row[col] = uuid.uuid4().int
            elif dtype == "object":
                row[col] = random.choice(["test", "train"])
            else:
                row[col] = None

        rows.append(row)

    return pd.DataFrame(rows)

datatypes = pd.read_csv("input_dataframe_datatypes.csv", sep=";").reset_index()

mock_df = generate_mock_df(datatypes, n_rows=500)

mock_df.to_csv("mock_input_data.csv", sep=";", index=False)