import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.exceptions import FitFailedWarning
import joblib
import warnings


def _configure_single_threaded():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JOBLIB_START_METHOD", "forkserver")


_configure_single_threaded()

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "mock_input_data.csv"

OUTPUT_BENCH = ROOT / "UKBRISK_ENModels" / "Benchmarking"
OUTPUT_NHC_INIT = ROOT / "UKBRISK_ENModels" / "NHC" / "Initial_10y"
OUTPUT_NHC_SEC = ROOT / "UKBRISK_ENModels" / "NHC" / "Secondary_10y"

PLOTS_ROOT = ROOT


def ensure_dirs():
    for d in [OUTPUT_BENCH, OUTPUT_NHC_INIT, OUTPUT_NHC_SEC]:
        d.mkdir(parents=True, exist_ok=True)

def read_mock_data():
    df = pd.read_csv(DATA_FILE, sep=";")
    # Drop leading unnamed column if present (from leading semicolon)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    if "" in df.columns:
        df = df.drop(columns=[""])
    return df


def split_train_test(df_filtered, labels, testtrain_column='testtrain'):
    
    train_data = df_filtered[df_filtered[testtrain_column] == 'train'].drop(columns=[testtrain_column])
    test_data = df_filtered[df_filtered[testtrain_column] == 'test'].drop(columns=[testtrain_column])

    train_labels = labels[labels[testtrain_column] == 'train'].drop(columns=[testtrain_column])
    test_labels = labels[labels[testtrain_column] == 'test'].drop(columns=[testtrain_column])

    return train_data, test_data, train_labels, test_labels


def train_opt_en(
    train_data,
    train_labels,
    l1_ratios=np.linspace(0.1, 1.0, 10),
    max_iter=100,
    alpha_min_ratio=0.01,
    cv_folds=5,
    verbose=True,
):

    labels_array = np.array(
        [(status, time) for status, time in zip(train_labels.iloc[:, 0], train_labels.iloc[:, 1])],
        dtype=[("event", "?"), ("time", "<f8")],
    )

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)

    print("estimating alphas with lambda=0.5...")

    initial_model = CoxnetSurvivalAnalysis(
        l1_ratio=0.5, alpha_min_ratio=alpha_min_ratio, max_iter=max_iter, n_alphas=5
    )
    initial_model.fit(train_data, labels_array)
    estimated_alphas = initial_model.alphas_


    print(
        f"estimated {len(estimated_alphas)} alphas ranging from {estimated_alphas.min():.5f} to {estimated_alphas.max():.5f}."
    )

    param_grid = {"l1_ratio": l1_ratios, "alphas": [[alpha] for alpha in estimated_alphas]}

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        CoxnetSurvivalAnalysis(max_iter=max_iter, fit_baseline_model=True),
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )

    grid_search.fit(train_data, labels_array)

    best_model = grid_search.best_estimator_
    best_l1_ratio = grid_search.best_params_["l1_ratio"]
    best_alpha = grid_search.best_params_["alphas"][0]

    if verbose:
        print(f"\nBest l1_ratio: {best_l1_ratio:.2f}, Best alpha: {best_alpha:.5f}")

    cv_results = pd.DataFrame(grid_search.cv_results_)

    return best_model, cv_results


def upload_model(model, endpoint, combo_name, cvresults, directory):
    filename_model = f"EN_{endpoint}_{combo_name}.pkl"
    filename_cvresults = f"EN_{endpoint}_{combo_name}_cvresults.tsv"

    joblib.dump(model, Path(directory) / filename_model)
    cvresults.to_csv(Path(directory) / filename_cvresults, sep="\t", index=False)


def save_and_upload_lps(model, train_data, test_data, train_labels, test_labels, endpoint, combo_name, directory):
    train_lp = model.predict(train_data)
    test_lp = model.predict(test_data)

    train_lp_df = pd.DataFrame({"eid": train_labels.index, "LP": train_lp})
    test_lp_df = pd.DataFrame({"eid": test_labels.index, "LP": test_lp})

    train_lp_filename = f"{endpoint}_{combo_name}_train_LP.tsv"
    test_lp_filename = f"{endpoint}_{combo_name}_test_LP.tsv"

    train_lp_df.to_csv(Path(directory) / train_lp_filename, sep="\t", index=False)
    test_lp_df.to_csv(Path(directory) / test_lp_filename, sep="\t", index=False)


def save_and_upload_coefficients(model, train_data, endpoint, combo_name, directory):
    coeff_filename = f"{endpoint}_{combo_name}_coefficients.tsv"
    coef_df = pd.DataFrame(model.coef_, index=train_data.columns, columns=["Coefficient"])
    coef_df.to_csv(Path(directory) / coeff_filename, sep="\t")


def calculate_and_upload_survival_probs(best_model, train_data, test_data, endpoint, combo_name, directory):
    unique_times = best_model.unique_times_
    time_point_index = (np.abs(unique_times - 10)).argmin()

    surv_probs_train = best_model.predict_survival_function(train_data, return_array=True)[:, time_point_index]
    surv_probs_test = best_model.predict_survival_function(test_data, return_array=True)[:, time_point_index]

    train_eid = train_data.index
    test_eid = test_data.index

    surv_10y_train_df = pd.DataFrame({"eid": train_eid, "survival_probability": surv_probs_train, "set": "train"})
    surv_10y_test_df = pd.DataFrame({"eid": test_eid, "survival_probability": surv_probs_test, "set": "test"})

    combined_df = pd.concat([surv_10y_train_df, surv_10y_test_df], ignore_index=True)

    filename_combined = f"EN_{endpoint}_{combo_name}_survival_probs_combined_10y.csv"
    combined_df.to_csv(Path(directory) / filename_combined, index=False)


def remove_low_count_logicals(df_filtered):
    logical_cols = df_filtered[
        [col for col in df_filtered.columns if (col.startswith("pmh_") or col.startswith("ts_")) and df_filtered[col].dtype == 'bool']
    ]
    if logical_cols.empty:
        return df_filtered
    cols_to_remove = [
        col for col in logical_cols.columns if logical_cols[col].mean() < 0.001 or logical_cols[col].mean() > 0.999
    ]
    return df_filtered.drop(columns=cols_to_remove)


def build_labels(df_filtered, endpoint):
    labels = df_filtered[[f"{endpoint}_status", f"{endpoint}_followup", "eid", "testtrain"]].copy()
    labels = labels.set_index("eid")
    return labels


def coerce_numeric(df):
    return df.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0})


def fit_benchmarking(df):
    endpoint_names = [
        "CVD", "HF", "BC", "DM", "LD", "RD", "AF", "CAD", "VT", "ISS", "AAA", "PAD",
        "AS", "COPD", "LC", "MEL", "CRC", "PC", "PD", "OP", "CAT", "POAG", "HT", "AD",
    ]

    always_include = ["clinicalrisk_Age.at.recruitment", "clinicalrisk_Sex_0", "clinicalrisk_Sex_1", "eid", "testtrain"]

    predictor_combinations = {
        "agesex": [],
        "pmh": ["pmh_"],
        "ts": ["ts_"],
        "metabolomics": ["metabolomics_"],
        "prs": ["prs_"],
        "clinicalrisk": ["clinicalrisk_"],
        "pmh_ts": ["pmh_", "ts_"],
        "prs_metabolomics": ["prs_", "metabolomics_"],
        "prs_metabolomics_pmh_ts": ["prs_", "metabolomics_", "pmh_", "ts_"],
        "clinicalrisk_pmh_ts": ["clinicalrisk_", "pmh_", "ts_"],
        "clinicalrisk_prs_metabolomics": ["clinicalrisk_", "prs_", "metabolomics_"],
        "everything": ["clinicalrisk_", "pmh_", "ts_", "prs_", "metabolomics_"],
        "score": ["score_"],
        "qrisk": ["qrisk_"],
        "prevent": ["prevent_"],
    }

    for endpoint in endpoint_names:
        if f"{endpoint}_status" not in df.columns:
            continue

        eids_to_include = df[df[f"{endpoint}_at_base"] == False]["eid"]
        df_filtered = df[df["eid"].isin(eids_to_include)]

        if endpoint == "PC" and "clinicalrisk_Sex_0" in df.columns:
            eids_to_exclude = df[df["clinicalrisk_Sex_0"] == True]["eid"]
            df_filtered = df_filtered[~df_filtered["eid"].isin(eids_to_exclude)]
        elif endpoint == "BC" and "clinicalrisk_Sex_1" in df.columns:
            eids_to_exclude = df[df["clinicalrisk_Sex_1"] == True]["eid"]
            df_filtered = df_filtered[~df_filtered["eid"].isin(eids_to_exclude)]

        df_filtered = remove_low_count_logicals(df_filtered)
        labels = build_labels(df_filtered, endpoint)

        for combo_name, prefixes in predictor_combinations.items():
            selected_cols = always_include + [
                col for col in df_filtered.columns if any(col.startswith(prefix) for prefix in prefixes) and col not in always_include
            ]
            df_filtered2 = df_filtered[selected_cols].set_index("eid")
            df_filtered2 = coerce_numeric(df_filtered2)

            train_data, test_data, train_labels, test_labels = split_train_test(df_filtered2, labels)
            if len(train_data) < 5 or train_labels.iloc[:, 0].sum() < 2:
                print(f"Skipping {endpoint} - {combo_name}: insufficient training data.")
                continue

            try:
                best_model, results_df = train_opt_en(train_data, train_labels)
            except Exception as exc:
                print(f"Skipping {endpoint} - {combo_name}: {exc}")
                continue

            upload_model(best_model, endpoint, combo_name, results_df, OUTPUT_BENCH)
            save_and_upload_lps(best_model, train_data, test_data, train_labels, test_labels, endpoint, combo_name, OUTPUT_BENCH)
            save_and_upload_coefficients(best_model, train_data, endpoint, combo_name, OUTPUT_BENCH)
            calculate_and_upload_survival_probs(best_model, train_data, test_data, endpoint, combo_name, OUTPUT_BENCH)


def fit_nhc_initial(df):
    endpoint_names_nhc = ["DM", "CVD", "RD"]

    always_include = ["clinicalrisk_Age.at.recruitment", "clinicalrisk_Sex_0", "clinicalrisk_Sex_1", "eid", "testtrain"]

    predictor_combinations = {
        "prs_metabolomics_pmh_ts": ["prs_", "metabolomics_", "pmh_", "ts_"],
        "pmh_ts": ["pmh_", "ts_"],
        "nhc": ["nhc_"],
        "nhc_pmh_ts": ["nhc", "pmh_", "ts_"],
        "nhc_prs_metabolomics_pmh_ts": ["nhc", "prs_", "metabolomics_", "pmh_", "ts_"],
    }

    for endpoint in endpoint_names_nhc:
        if f"{endpoint}_status" not in df.columns:
            continue

        eids_to_include = df[df[f"{endpoint}_at_base"] == False]["eid"]
        df_filtered = df[df["eid"].isin(eids_to_include)]

        df_filtered = remove_low_count_logicals(df_filtered)
        labels = build_labels(df_filtered, endpoint)

        for combo_name, prefixes in predictor_combinations.items():
            selected_cols = always_include + [
                col for col in df_filtered.columns if any(col.startswith(prefix) for prefix in prefixes) and col not in always_include
            ]
            df_filtered2 = df_filtered[selected_cols].set_index("eid")
            df_filtered2 = coerce_numeric(df_filtered2)

            train_data, test_data, train_labels, test_labels = split_train_test(df_filtered2, labels)
            if len(train_data) < 5 or train_labels.iloc[:, 0].sum() < 2:
                print(f"Skipping NHC initial {endpoint} - {combo_name}: insufficient training data.")
                continue

            try:
                best_model, results_df = train_opt_en(train_data, train_labels)
            except Exception as exc:
                print(f"Skipping NHC initial {endpoint} - {combo_name}: {exc}")
                continue

            upload_model(best_model, endpoint, combo_name, results_df, OUTPUT_NHC_INIT)
            save_and_upload_lps(best_model, train_data, test_data, train_labels, test_labels, endpoint, combo_name, OUTPUT_NHC_INIT)
            save_and_upload_coefficients(best_model, train_data, endpoint, combo_name, OUTPUT_NHC_INIT)
            calculate_and_upload_survival_probs(best_model, train_data, test_data, endpoint, combo_name, OUTPUT_NHC_INIT)


def build_absrisk_frames(directory):
    absrisk_files = [p.name for p in Path(directory).glob("*_combined_10y.csv")]
    merged_dataframes = {}

    for file_name in absrisk_files:
        abs_risk_data = pd.read_csv(Path(directory) / file_name)
        parts = file_name.split("_")
        endpoint = parts[1]
        survival_index = parts.index("survival")
        combo_name = "_".join(parts[2:survival_index])

        if endpoint in merged_dataframes:
            merged_dataframes[endpoint] = pd.merge(
                merged_dataframes[endpoint],
                abs_risk_data[["eid", "survival_probability", "set"]],
                on=["eid", "set"],
                how="outer",
            )
            merged_dataframes[endpoint].rename(columns={"survival_probability": combo_name}, inplace=True)
        else:
            abs_risk_data_merged = abs_risk_data[["eid", "set", "survival_probability"]]
            abs_risk_data_merged.rename(columns={"survival_probability": combo_name}, inplace=True)
            merged_dataframes[endpoint] = abs_risk_data_merged

    train_frames = {k: v[v["set"] == "train"].copy() for k, v in merged_dataframes.items()}
    test_frames = {k: v[v["set"] == "test"].copy() for k, v in merged_dataframes.items()}

    return train_frames, test_frames


def fit_nhc_secondary(df):
    endpoint_names_nhc = ["DM", "CVD", "RD"]

    always_include = ["clinicalrisk_Age.at.recruitment", "clinicalrisk_Sex_0", "clinicalrisk_Sex_1", "eid", "testtrain"]

    predictor_combinations = {
        "nhc_pmh_ts": ["nhc", "pmh_", "ts_"],
        "nhc_prs_metabolomics_pmh_ts": ["nhc", "prs_", "metabolomics_", "pmh_", "ts_"],
    }

    risk_columns = ["pmh_ts", "prs_metabolomics_pmh_ts"]

    train_abs, test_abs = build_absrisk_frames(OUTPUT_NHC_INIT)

    for risk_col in risk_columns:
        cvd_risk_filtered_5_train = train_abs["CVD"].loc[train_abs["CVD"][risk_col] < 0.95, "eid"]
        rd_risk_filtered_5_train = train_abs["RD"].loc[train_abs["RD"][risk_col] < 0.95, "eid"]
        dm_risk_filtered_5_train = train_abs["DM"].loc[train_abs["DM"][risk_col] < 0.95, "eid"]

        high_risk_eids_5_train = set(cvd_risk_filtered_5_train).union(set(rd_risk_filtered_5_train), set(dm_risk_filtered_5_train))

        cvd_risk_filtered_10_train = train_abs["CVD"].loc[train_abs["CVD"][risk_col] < 0.90, "eid"]
        rd_risk_filtered_10_train = train_abs["RD"].loc[train_abs["RD"][risk_col] < 0.90, "eid"]
        dm_risk_filtered_10_train = train_abs["DM"].loc[train_abs["DM"][risk_col] < 0.90, "eid"]

        high_risk_eids_10_train = set(cvd_risk_filtered_10_train).union(set(rd_risk_filtered_10_train), set(dm_risk_filtered_10_train))

        cvd_risk_filtered_5_test = test_abs["CVD"].loc[test_abs["CVD"][risk_col] < 0.95, "eid"]
        rd_risk_filtered_5_test = test_abs["RD"].loc[test_abs["RD"][risk_col] < 0.95, "eid"]
        dm_risk_filtered_5_test = test_abs["DM"].loc[test_abs["DM"][risk_col] < 0.95, "eid"]

        high_risk_eids_5_test = set(cvd_risk_filtered_5_test).union(set(rd_risk_filtered_5_test), set(dm_risk_filtered_5_test))

        cvd_risk_filtered_10_test = test_abs["CVD"].loc[test_abs["CVD"][risk_col] < 0.90, "eid"]
        rd_risk_filtered_10_test = test_abs["RD"].loc[test_abs["RD"][risk_col] < 0.90, "eid"]
        dm_risk_filtered_10_test = test_abs["DM"].loc[test_abs["DM"][risk_col] < 0.90, "eid"]

        high_risk_eids_10_test = set(cvd_risk_filtered_10_test).union(set(rd_risk_filtered_10_test), set(dm_risk_filtered_10_test))

        for endpoint in endpoint_names_nhc:
            if f"{endpoint}_status" not in df.columns:
                continue

            eids_to_include = df[df[f"{endpoint}_at_base"] == False]["eid"]
            df_filtered = df[df["eid"].isin(eids_to_include)]

            df_filtered = remove_low_count_logicals(df_filtered)
            labels = build_labels(df_filtered, endpoint)

            for combo_name, prefixes in predictor_combinations.items():
                selected_cols = always_include + [
                    col for col in df_filtered.columns if any(col.startswith(prefix) for prefix in prefixes) and col not in always_include
                ]
                df_filtered2 = df_filtered[selected_cols].set_index("eid")
                df_filtered2 = coerce_numeric(df_filtered2)

                train_data, test_data, train_labels, test_labels = split_train_test(df_filtered2, labels)

                train_data_5 = train_data[train_data.index.isin(high_risk_eids_5_train)]
                train_data_10 = train_data[train_data.index.isin(high_risk_eids_10_train)]

                test_data_5 = test_data[test_data.index.isin(high_risk_eids_5_test)]
                test_data_10 = test_data[test_data.index.isin(high_risk_eids_10_test)]

                if not train_data_5.empty and not test_data_5.empty:
                    model_suffix_5pct = f"risk_{risk_col}_model_{combo_name}_5pct"
                    try:
                        best_model_5, results_df_5 = train_opt_en(train_data_5, train_labels.loc[train_data_5.index])
                    except Exception as exc:
                        print(f"Skipping NHC secondary 5% {endpoint} - {combo_name}: {exc}")
                        continue
                    upload_model(best_model_5, endpoint, model_suffix_5pct, results_df_5, OUTPUT_NHC_SEC)
                    save_and_upload_lps(
                        best_model_5,
                        train_data_5,
                        test_data_5,
                        train_labels.loc[train_data_5.index],
                        test_labels.loc[test_data_5.index],
                        endpoint,
                        model_suffix_5pct,
                        OUTPUT_NHC_SEC,
                    )
                    save_and_upload_coefficients(best_model_5, train_data_5, endpoint, model_suffix_5pct, OUTPUT_NHC_SEC)
                    calculate_and_upload_survival_probs(best_model_5, train_data_5, test_data_5, endpoint, model_suffix_5pct, OUTPUT_NHC_SEC)

                if not train_data_10.empty and not test_data_10.empty:
                    model_suffix_10pct = f"risk_{risk_col}_model_{combo_name}_10pct"
                    try:
                        best_model_10, results_df_10 = train_opt_en(train_data_10, train_labels.loc[train_data_10.index])
                    except Exception as exc:
                        print(f"Skipping NHC secondary 10% {endpoint} - {combo_name}: {exc}")
                        continue
                    upload_model(best_model_10, endpoint, model_suffix_10pct, results_df_10, OUTPUT_NHC_SEC)
                    save_and_upload_lps(
                        best_model_10,
                        train_data_10,
                        test_data_10,
                        train_labels.loc[train_data_10.index],
                        test_labels.loc[test_data_10.index],
                        endpoint,
                        model_suffix_10pct,
                        OUTPUT_NHC_SEC,
                    )
                    save_and_upload_coefficients(best_model_10, train_data_10, endpoint, model_suffix_10pct, OUTPUT_NHC_SEC)
                    calculate_and_upload_survival_probs(best_model_10, train_data_10, test_data_10, endpoint, model_suffix_10pct, OUTPUT_NHC_SEC)


def create_plot_inputs(df):
    plot_records = []
    baseline_records = {}

    lp_files = list(OUTPUT_BENCH.glob("*_test_LP.tsv"))
    for lp_file in lp_files:
        parts = lp_file.name.split("_")
        endpoint = parts[0]
        combo_name = "_".join(parts[1:-2])

        if f"{endpoint}_status" not in df.columns:
            continue

        labels = df[["eid", f"{endpoint}_status", f"{endpoint}_followup"]].copy()
        labels = labels.set_index("eid")

        lp_df = pd.read_csv(lp_file, sep="\t")
        merged = labels.join(lp_df.set_index("eid"), how="inner")
        if merged.empty:
            continue

        status = merged[f"{endpoint}_status"].astype(bool).to_numpy()
        time = merged[f"{endpoint}_followup"].to_numpy()
        lp = merged["LP"].to_numpy()
        try:
            c_index = concordance_index_censored(status, time, -lp)[0]
        except Exception:
            continue

        plot_records.append(
            {
                "Endpoint": endpoint,
                "Model": combo_name,
                "C_Index": c_index,
                "Bottom95": c_index,
                "Top95": c_index,
            }
        )
        if combo_name == "clinicalrisk":
            baseline_records[endpoint] = c_index

    plot_data_abs = pd.DataFrame(plot_records)
    plot_data_abs.to_csv(PLOTS_ROOT / "absC_benchmarking_bootstrapped.tsv", sep="\t", index=False)

    delta_records = []
    for rec in plot_records:
        endpoint = rec["Endpoint"]
        model = rec["Model"]
        if endpoint not in baseline_records or model == "clinicalrisk":
            continue
        delta = rec["C_Index"] - baseline_records[endpoint]
        delta_records.append(
            {
                "Endpoint": endpoint,
                "Model2": model,
                "MeanDeltaC": delta,
                "Bottom95": delta,
                "Top95": delta,
            }
        )

    plot_data_delta = pd.DataFrame(delta_records)
    plot_data_delta.to_csv(PLOTS_ROOT / "deltaC_benchmarking_bootstrapped.tsv", sep="\t", index=False)

def main():
    ensure_dirs()
    df = read_mock_data()

    df.loc[df["CVD_followup"] > 10, "CVD_status"] = False
    df.loc[df["RD_followup"] > 10, "RD_status"] = False
    df.loc[df["DM_followup"] > 10, "DM_status"] = False

    df["CVD_followup"] = df["CVD_followup"].clip(upper=10)
    df["RD_followup"] = df["RD_followup"].clip(upper=10)
    df["DM_followup"] = df["DM_followup"].clip(upper=10)

    fit_benchmarking(df)
    fit_nhc_initial(df)
    fit_nhc_secondary(df)

    create_plot_inputs(df)

if __name__ == "__main__":
    main()
