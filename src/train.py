import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    log_loss,
    classification_report,
    f1_score,
    accuracy_score
)
#ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# OOF TARGET ENCODING


def oof_encode(df, column, target, smoothing):
    df = df.copy()
    encoded = pd.Series(index=df.index, dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = df[target].mean()

    for train_idx, val_idx in skf.split(df, df[target]):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        stats = train.groupby(column)[target].agg(["mean", "count"])
        smooth = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )

        encoded.iloc[val_idx] = val[column].map(smooth)

    encoded.fillna(global_mean, inplace=True)
    return encoded



# DATA PREPARATION


def load_and_prepare_data(path):

    DATA = pd.read_csv(path)

    # Drop irrelevant columns
    DATA = DATA.drop(columns=[
        "Transaction Notes",
        "Cardholder Name",
        "Card Number (Hashed or Encrypted)",
        "CVV Code (Hashed or Encrypted)",
        "Transaction ID",
        "IP Address"
    ])

    # Handle missing values
    imputer_num = SimpleImputer(strategy="mean")
    imputer_cat = SimpleImputer(strategy="most_frequent")

    category_cols = DATA.drop(
        columns=["Transaction Amount", "Fraud Flag or Label"]
    ).columns.tolist()

    DATA["Transaction Amount"] = imputer_num.fit_transform(
        DATA[["Transaction Amount"]]
    )

    DATA[category_cols] = imputer_cat.fit_transform(DATA[category_cols])

    # Sort for time-series consistency
    DATA = DATA.sort_values(
        ["User Account Information", "Transaction Date and Time"]
    )

    # Expanding mean feature
    DATA["Transaction Amount_mean"] = (
        DATA.groupby("User Account Information")["Transaction Amount"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    Data = pd.DataFrame()

    Data["Transaction Amount_mean"] = (
        DATA.groupby("User Account Information")["Transaction Amount_mean"]
        .last()
    )

    # OOF encoding
    encoder_cat = {
        "Merchant Name": 7.12,
        "Merchant Category Code (MCC)": 14.17,
        "Card Expiration Date": 6.09,
        "Transaction Location (City or ZIP Code)": 8.10
    }

    for col, smoothing in encoder_cat.items():
        DATA[col + "_oof"] = oof_encode(
            DATA, column=col,
            target="Fraud Flag or Label",
            smoothing=smoothing
        )

        Data[col + "_oof"] = (
            DATA.groupby("User Account Information")[col + "_oof"]
            .last()
        )

    # Small categorical features
    ohe_cols = DATA.select_dtypes(include=["object"]).columns.tolist()
    drop_cols = list(encoder_cat.keys()) + [
        "User Account Information",
        "Transaction Date and Time"
    ]
    ohe_cols = [c for c in ohe_cols if c not in drop_cols]

    for col in ohe_cols:
        Data[col] = (
            DATA.groupby("User Account Information")[col]
            .last()
            .astype("category")
        )

    # Target
    Y = (
        DATA.groupby("User Account Information")
        ["Fraud Flag or Label"]
        .max()
        .fillna(0)
        .astype(int)
    )

    return Data, Y, ohe_cols



# THRESHOLD TUNING


def find_best_threshold(y_true, y_prob):

    thresholds = np.linspace(0.1, 0.9, 100)

    best_f1 = 0
    best_acc = 0
    best_f1_thresh = 0.5
    best_acc_thresh = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t

        if acc > best_acc:
            best_acc = acc
            best_acc_thresh = t

    return best_f1_thresh, best_f1, best_acc_thresh, best_acc



# MAIN TRAINING PIPELINE


def main():
    # Make sure the dataset is in the same directory
    X, Y, categorical_cols = load_and_prepare_data(
        os.path.join("data", "credit_card_fraud_test1.csv")
             )
    

    x_train, x_temp, y_train, y_temp = train_test_split(
        X, Y, train_size=0.6,
        random_state=46,
        stratify=Y
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        train_size=0.5,
        random_state=46,
        stratify=y_temp
    )

    ratio = sum(y_train == 0) / sum(y_train == 1)

    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_cols
    )

    valid_data = lgb.Dataset(
        x_val,
        label=y_val,
        reference=train_data,
        categorical_feature=categorical_cols
    )

    def objective(trial):

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 20, 100
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10, log=True
            ),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": ratio,
            "feature_pre_filter":False
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(30)]
        )

        preds = model.predict(
            x_val,
            num_iteration=model.best_iteration
        )

        return log_loss(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best LogLoss:", study.best_value)

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": ratio
    })

    model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30)]
    )

    y_prob = model.predict(
        x_test,
        num_iteration=model.best_iteration
    )

    print("\nLog Loss:", log_loss(y_test, y_prob))

    # Threshold tuning
    f1_t, f1_val, acc_t, acc_val = find_best_threshold(
        y_test, y_prob
    )

    print("\nBest F1 Threshold:", f1_t)
    print("Best F1 Score:", f1_val)

    print("\nBest Accuracy Threshold:", acc_t)
    print("Best Accuracy:", acc_val)

    # Final report (F1 optimized)
    y_pred = (y_prob >= f1_t).astype(int)

    print("\nClassification Report (F1 Optimized):")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
