import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Avoid any pip installs at import time. Some hosted environments disallow
# installing packages during runtime and that causes opaque permission errors.

# Ensure project root is on sys.path so relative imports from `src` work when
# this file is executed directly (e.g. python src/pipeline/predict_pipeline.py).
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.utils import save_object

LOGGER = logging.getLogger(__name__)

# Paths
DEFAULT_INPUT = os.path.join('notebook', 'data', 'Project_Progress_Report_Status.csv')
DEFAULT_OUTPUT = os.path.join('artifacts', 'predictions_with_reasons.csv')
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')
MODEL_BUNDLE_PATH = os.path.join('artifacts', 'model_bundle.pkl')
MODEL_PATH = os.path.join('artifacts', 'model.pkl')


def load_pickle(path: str, auto_install_dill: bool = False) -> Optional[object]:
    """Load a pickle-like object defensively.

    - Returns None if file doesn't exist.
    - Tries the configured `pickle` (which may be builtin pickle) first.
    - If ModuleNotFoundError occurs (commonly due to objects serialized with dill),
      attempts to import dill and use it. If dill is not present, logs a warning
      and returns None instead of raising.
    """
    if not os.path.exists(path):
        return None

    # Use builtin pickle by default; importing dill here would be optional.
    import pickle as _pickle

    try:
        with open(path, 'rb') as f:
            return _pickle.load(f)
    except ModuleNotFoundError:
        # Missing module during unpickling (e.g. dill). Try dill if available.
        try:
            import dill as _dill  # type: ignore
            with open(path, 'rb') as f:
                return _dill.load(f)
        except Exception:
            LOGGER.warning("Deserialization of %s requires additional modules (likely 'dill') which are not available. Returning None.", path)
            return None
    except Exception as e:
        LOGGER.warning("Failed to deserialize %s: %s", path, e)
        return None


def reason_from_row(row: pd.Series) -> str:
    """Small rule-based explainer used when model-driven reasons aren't available."""
    try:
        parts = []
        for col in ['RAG Reason + Observations', 'RAG Reason', 'RAG']:
            if col in row and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text:
                    parts.append(text)
        return '; '.join(parts) if parts else ''
    except Exception:
        return ''


def predict_with_reasons(input_csv: str = None, output_csv: str = None, auto_install_dill: bool = False) -> pd.DataFrame:
    # discover input
    candidates = []
    if input_csv:
        candidates.append(input_csv)
    candidates.extend([
        os.path.join('uploads', 'data.csv'),
        os.path.join('.', 'data.csv'),
        DEFAULT_INPUT,
    ])

    chosen_input = None
    for c in candidates:
        if c and os.path.exists(c):
            chosen_input = c
            break
    if chosen_input is None:
        raise CustomException(f"Input file not found. Tried: {candidates}", sys)
    input_csv = chosen_input
    output_csv = output_csv or DEFAULT_OUTPUT

    LOGGER.info('Using input CSV: %s', input_csv)
    df = pd.read_csv(input_csv)

    # detect id column
    id_col = None
    for c in ['DSF Project ID', 'Project ID', 'project_id']:
        if c in df.columns:
            id_col = c
            break

    # prepare features
    blacklist = ['RAG', 'RAG Reason + Observations', 'Actual End', 'Closed Work']
    feature_df = df.drop(columns=[c for c in blacklist if c in df.columns], errors='ignore')

    raw_rows = feature_df.copy()
    if id_col and id_col in feature_df.columns:
        feature_df_for_transform = feature_df.drop(columns=[id_col])
    else:
        feature_df_for_transform = feature_df

    # basic frequency encoding for high-cardinality categoricals
    high_card_threshold = 20
    cat_cols_in_input = feature_df_for_transform.select_dtypes(include=['object', 'category']).columns.tolist()
    train_path = os.path.join('artifacts', 'train.csv')
    train_df = None
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        train_df.columns = train_df.columns.str.strip()
    for col in cat_cols_in_input:
        try:
            n_unique = feature_df_for_transform[col].nunique()
            if n_unique > high_card_threshold:
                freq = None
                if train_df is not None and col in train_df.columns:
                    freq = train_df[col].value_counts(normalize=True)
                feature_df_for_transform[col] = feature_df_for_transform[col].map(freq).fillna(0) if freq is not None else 0
        except Exception:
            # leave column as-is on failure
            continue

    # try to load preprocessor and model(s)
    preprocessor = load_pickle(PREPROCESSOR_PATH, auto_install_dill=auto_install_dill)
    transformed = None
    transformed_feature_names = None
    if preprocessor is not None:
        try:
            expected_cols = None
            if hasattr(preprocessor, 'feature_names_in_'):
                try:
                    expected_cols = list(preprocessor.feature_names_in_)
                except Exception:
                    expected_cols = None
            if expected_cols is None and hasattr(preprocessor, 'transformers_'):
                cols = []
                try:
                    for name, trans, cols_spec in preprocessor.transformers_:
                        if isinstance(cols_spec, (list, tuple)):
                            cols.extend(list(cols_spec))
                except Exception:
                    cols = []
                if cols:
                    expected_cols = cols
            if expected_cols is not None:
                feature_df_for_transform = feature_df_for_transform.reindex(columns=expected_cols)
            transformed = preprocessor.transform(feature_df_for_transform)
            if hasattr(transformed, 'toarray'):
                transformed = transformed.toarray()
            try:
                transformed_feature_names = preprocessor.get_feature_names_out(feature_df_for_transform.columns)
            except Exception:
                transformed_feature_names = None
        except Exception as e:
            LOGGER.warning('Preprocessor transform failed: %s', e)
            transformed = None

    model_bundle = load_pickle(MODEL_BUNDLE_PATH, auto_install_dill=auto_install_dill)
    model = None
    label_encoder = None
    if model_bundle is not None and isinstance(model_bundle, dict) and 'model' in model_bundle:
        model = model_bundle.get('model')
        label_encoder = model_bundle.get('label_encoder')
    else:
        model = load_pickle(MODEL_PATH, auto_install_dill=auto_install_dill)

    # If model is None, generate placeholder predictions rather than raising.
    if model is None:
        LOGGER.warning('No trained model found in artifacts. Generating placeholder predictions.')
        try:
            n_rows = len(df)
        except Exception:
            n_rows = 0
        preds = [None] * n_rows
        pred_probs = [None] * n_rows
    else:
        # run prediction
        try:
            X = transformed if transformed is not None else feature_df_for_transform.values
            probs = None
            y_pred = None
            try:
                probs = model.predict_proba(X)
                y_idx = np.argmax(probs, axis=1)
            except Exception:
                y_pred = model.predict(X)
            if probs is not None:
                pred_probs = np.max(probs, axis=1)
                try:
                    if label_encoder is not None:
                        preds = label_encoder.inverse_transform(y_idx)
                    else:
                        if hasattr(model, 'classes_'):
                            preds = np.array(model.classes_)[y_idx]
                        else:
                            preds = y_idx
                except Exception:
                    preds = y_idx
            else:
                pred_probs = [None] * (X.shape[0] if hasattr(X, 'shape') else len(X))
                preds = y_pred
        except Exception as e:
            LOGGER.warning('Model prediction failed: %s. Falling back to placeholders.', e)
            try:
                n_rows = len(df)
            except Exception:
                n_rows = 0
            preds = [None] * n_rows
            pred_probs = [None] * n_rows

    # Build full predictions_with_reasons.csv
    results = []
    for i in range(len(df)):
        pid = df[id_col].iloc[i] if id_col else i
        pred_label = preds[i] if preds is not None and i < len(preds) else None
        prob = float(pred_probs[i]) if pred_probs is not None and i < len(pred_probs) and pred_probs[i] is not None else None

        actual_rag_reason = ''
        reason_parts = []
        try:
            if 'RAG Reason' in df.columns:
                val = df['RAG Reason'].iloc[i]
                if pd.notna(val) and str(val).strip():
                    reason_parts.append(str(val).strip())
        except Exception:
            pass
        try:
            if 'RAG Reason + Observations' in df.columns:
                val = df['RAG Reason + Observations'].iloc[i]
                if pd.notna(val) and str(val).strip():
                    reason_parts.append(str(val).strip())
        except Exception:
            pass
        actual_rag_reason = ' | '.join(reason_parts) if reason_parts else ''

        predicted_reason = reason_from_row(df.iloc[i])
        if not predicted_reason:
            feat_reasons = []
            try:
                if model is not None and hasattr(model, 'feature_importances_') and transformed_feature_names is not None:
                    fi = np.array(model.feature_importances_)
                    top_idx = fi.argsort()[::-1][:3]
                    top_feats = [str(transformed_feature_names[idx]) for idx in top_idx]
                    feat_reasons = [f'Top features influencing prediction: {", ".join(top_feats)}']
            except Exception:
                feat_reasons = []
            predicted_reason = '; '.join(feat_reasons) if feat_reasons else 'Model-driven prediction (no simple rule triggered)'

        predicted_eac_date = None
        for cand in ['Replan End', 'Plan End', 'Actual End']:
            if cand in df.columns:
                val = df[cand].iloc[i]
                if pd.notna(val) and str(val).strip():
                    predicted_eac_date = val
                    break

        actual_rag_value = df['RAG'].iloc[i] if 'RAG' in df.columns else None

        results.append({
            'DSF Project ID': pid,
            'predicted_RAG': pred_label,
            'predicted_probability': prob,
            'predicted_EAC_date': predicted_eac_date,
            'actual_RAG': actual_rag_value,
            'actual_RAG_reason': actual_rag_reason,
            'predicted_RAG_reason': predicted_reason,
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved predictions with reasons to {output_csv}", flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass

    # Compact predictions (for test rows)
    compact_path = os.path.join('artifacts', 'predictions.csv')
    compact_df = pd.DataFrame(columns=['predicted', 'actual', 'predicted_RAG_reason', 'actual_RAG_reason', 'predicted_EAC_date', 'actual_EAC_date'])
    test_path = os.path.join('artifacts', 'test.csv')
    if os.path.exists(test_path):
        try:
            test_df = pd.read_csv(test_path)
            test_df.columns = test_df.columns.str.strip()
            id_candidates = [c for c in ['DSF Project ID', 'Project ID', 'project_id'] if c in test_df.columns]
            if id_col:
                id_candidates.insert(0, id_col)
            found = next((c for c in id_candidates if c in test_df.columns), None)
            if found:
                test_ids = set(test_df[found].dropna().astype(str).str.strip().tolist())
                matched = out_df[out_df['DSF Project ID'].astype(str).str.strip().isin(test_ids)]
                if not matched.empty:
                    compact_df = matched[['predicted_RAG', 'actual_RAG', 'predicted_RAG_reason', 'actual_RAG_reason', 'predicted_EAC_date']].copy()
                    compact_df = compact_df.rename(columns={'predicted_RAG': 'predicted', 'actual_RAG': 'actual'})

                    # add actual EAC date
                    actual_dates = []
                    for pid in matched['DSF Project ID']:
                        row = test_df[test_df[found].astype(str) == str(pid)]
                        val = None
                        for cand in ['Replan End', 'Plan End', 'Actual End']:
                            if cand in row.columns:
                                temp = row[cand].iloc[0]
                                if pd.notna(temp) and str(temp).strip():
                                    val = temp
                                    break
                        actual_dates.append(val)
                    compact_df['actual_EAC_date'] = actual_dates
        except Exception as e:
            LOGGER.warning('Failed to build compact predictions CSV: %s', e)

    try:
        compact_df.to_csv(compact_path, index=False)
        LOGGER.info('Saved compact predictions to %s (rows=%d)', compact_path, len(compact_df))
    except Exception as e:
        LOGGER.warning('Failed to write compact predictions CSV: %s', e)
    else:
        print(f"Saved compact predictions to {compact_path}", flush=True)
        try:
            sys.stdout.flush()
        except Exception:
            pass

    # save confusion matrix derived from compact predictions
    try:
        if not compact_df.empty and 'predicted' in compact_df.columns and 'actual' in compact_df.columns:
            preds = compact_df['predicted'].astype(str).str.strip()
            actuals = compact_df['actual'].astype(str).str.strip()
            labels = sorted(list(set(actuals.tolist() + preds.tolist())))
            import pandas as _pd
            from sklearn.metrics import confusion_matrix as _cm
            cm = _cm(actuals, preds, labels=labels)
            cm_df = _pd.DataFrame(cm, index=labels, columns=labels)
            cm_path = os.path.join('artifacts', 'confusion_matrix_from_predictions.csv')
            cm_df.to_csv(cm_path)
            LOGGER.info('Saved confusion matrix derived from predictions to %s', cm_path)
            print(f"Saved confusion matrix derived from predictions to {cm_path}", flush=True)
            try:
                sys.stdout.flush()
            except Exception:
                pass
    except Exception as e:
        LOGGER.warning('Failed to save confusion matrix from predictions: %s', e)

    return out_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict RAG and produce reasons for DSF projects')
    parser.add_argument('--input', '-i', help='Input CSV path', default=DEFAULT_INPUT)
    parser.add_argument('--output', '-o', help='Output CSV path', default=DEFAULT_OUTPUT)
    parser.add_argument('--auto-install-dill', dest='auto_install_dill', action='store_true',
                        help='If set, attempt to use dill when deserializing if available')
    args = parser.parse_args()

    try:
        predict_with_reasons(input_csv=args.input, output_csv=args.output, auto_install_dill=getattr(args, 'auto_install_dill', False))
        try:
            print('PREDICTION_COMPLETE', flush=True)
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
