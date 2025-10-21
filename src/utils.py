import os
import sys

import numpy as np
import pandas as pd
# Prefer dill if available (better at serializing complex objects); fall back to
# the stdlib pickle if dill is not installed so scripts work in minimal envs.
try:
    import dill as _pickle_lib
except Exception:
    import pickle as _pickle_lib
from sklearn.metrics import r2_score, accuracy_score

from src.exception import CustomException
from sklearn.preprocessing import LabelEncoder


def find_input_file(preferred_subdir='notebook/data'):
    """Discover an input file to use for ingestion/prediction.

    Search order:
    1. Date-prefixed files in `notebook/data` (YYYY-MM-DD*)
    2. Any CSV/XLSX in `notebook/data` (newest)
    3. Any CSV/XLSX in `uploads` (newest)
    4. './data.csv' in project root

    Returns absolute path or None if not found.
    """
    try:
        nb_dir = os.path.join(*preferred_subdir.split('/'))
        candidates = []
        if os.path.isdir(nb_dir):
            for fname in os.listdir(nb_dir):
                if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
                    candidates.append(os.path.join(nb_dir, fname))
        if candidates:
            import re
            date_pref = [p for p in candidates if re.match(r"^\d{4}-\d{2}-\d{2}", os.path.basename(p))]
            if date_pref:
                return os.path.abspath(max(date_pref, key=os.path.getmtime))
            return os.path.abspath(max(candidates, key=os.path.getmtime))

        # fallback to uploads
        uploads_dir = 'uploads'
        uploads_candidates = []
        if os.path.isdir(uploads_dir):
            for fname in os.listdir(uploads_dir):
                if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
                    uploads_candidates.append(os.path.join(uploads_dir, fname))
        if uploads_candidates:
            return os.path.abspath(max(uploads_candidates, key=os.path.getmtime))

        # fallback to data.csv
        if os.path.exists('data.csv'):
            return os.path.abspath('data.csv')

        return None
    except Exception:
        return None

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            _pickle_lib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Flatten target arrays to 1D
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()

        # Detect if this is a classification task: original dtype object or strings
        is_classification = False
        if y_train.dtype == object or y_test.dtype == object:
            is_classification = True
            le = LabelEncoder()
            # Fit on combined to keep consistent mapping
            le.fit(np.concatenate([y_train, y_test]))
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)

        # Drop rows with NaNs in either X or y for train and test splits
        def drop_nan_rows(X, y):
            X = np.array(X)
            y = np.array(y)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Build mask for finite rows. For non-numeric columns, isnan will raise; handle safely.
            try:
                nan_mask_X = np.isnan(X).any(axis=1)
            except Exception:
                # If X has non-numeric types, treat them as non-NaN (can't detect)
                nan_mask_X = np.zeros(X.shape[0], dtype=bool)

            try:
                nan_mask_y = np.isnan(y)
            except Exception:
                nan_mask_y = np.zeros(y.shape[0], dtype=bool)

            mask = ~(nan_mask_X | nan_mask_y)
            return X[mask], y[mask]

        # for safety, cast to float where possible for NaN checks
        try:
            X_train = X_train.astype(float)
        except Exception:
            pass
        try:
            X_test = X_test.astype(float)
        except Exception:
            pass

        X_train, y_train = drop_nan_rows(X_train, y_train)
        X_test, y_test = drop_nan_rows(X_test, y_test)

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            if is_classification:
                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
            else:
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        # Return report and processed test arrays + encoding info so callers can use the same labels
        return report, X_test, y_test, is_classification, (le if is_classification else None)

    except Exception as e:
        raise CustomException(e, sys)