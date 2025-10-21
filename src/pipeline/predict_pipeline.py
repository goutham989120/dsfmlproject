import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# When this script is executed directly (python src/pipeline/predict_pipeline.py)
# the interpreter's sys.path may not include the project root, which prevents
# imports like `from src.exception import CustomException` from working.
# Ensure the project root (two levels up from this file) is on sys.path.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    # insert at front so local packages take precedence
    sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.utils import save_object

try:
    import dill as pickle
except Exception:
    import pickle

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = os.path.join('notebook', 'data', 'Project_Progress_Report_Status.csv')
DEFAULT_OUTPUT = os.path.join('artifacts', 'predictions_with_reasons.csv')
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')
MODEL_BUNDLE_PATH = os.path.join('artifacts', 'model_bundle.pkl')
MODEL_PATH = os.path.join('artifacts', 'model.pkl')


def load_pickle(path: str, auto_install_dill: bool = False):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as mnf:
            # The pickle.load failed because the file references a module
            # that's not available in the environment (commonly 'dill').
            # Try to import dill and use it; if dill isn't installed, raise a
            # helpful CustomException instructing the user to install it.
            try:
                import dill as _dill
            except Exception as ie:
                # Optionally attempt to auto-install dill if requested via flag or env var
                auto_env = os.environ.get('DSFML_AUTO_INSTALL_DILL', '').lower() in ('1', 'true', 'yes')
                if auto_install_dill or auto_env:
                    try:
                        import subprocess
                        cmd = [sys.executable, '-m', 'pip', 'install', 'dill']
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
                        if proc.returncode != 0:
                            # If pip failed because of permission issues, avoid raising
                            # a verbose error that repeatedly shows up in the Streamlit UI.
                            out = proc.stdout or ''
                            low = out.lower()
                            if 'permission denied' in low or 'errno 13' in low or "could not install packages due to an oserror" in low:
                                LOGGER.warning("Auto-install of 'dill' failed due to permissions; continuing without dill. Output: %s", out)
                                return None
                            raise Exception(f"pip install returned {proc.returncode}: {proc.stdout}")
                        # try importing again
                        import importlib
                        importlib.invalidate_caches()
                        import dill as _dill
                    except Exception as ie2:
                        raise CustomException(
                            f"Failed to auto-install 'dill' while deserializing {path}.\n"
                            f"Tried: {' '.join(cmd)}\n"
                            f"Error: {ie2}\n"
                            f"Please install dill manually: pip install dill",
                            sys,
                        ) from ie2
                else:
                    raise CustomException(
                        f"Failed to deserialize {path}: the file requires 'dill' but the package is not installed.\n"
                        f"Install it with: pip install dill",
                        sys,
                    ) from ie
            try:
                f.seek(0)
                return _dill.load(f)
            except Exception as e:
                raise CustomException(e, sys)
        except Exception as e:
            # Generic fallback: the object might still require dill to
            # deserialize. Try dill if available, otherwise wrap the error.
            try:
                import dill as _dill
                f.seek(0)
                return _dill.load(f)
            except ModuleNotFoundError:
                # If auto-install requested, attempt install then retry
                auto_env = os.environ.get('DSFML_AUTO_INSTALL_DILL', '').lower() in ('1', 'true', 'yes')
                if auto_install_dill or auto_env:
                    try:
                        import subprocess
                        cmd = [sys.executable, '-m', 'pip', 'install', 'dill']
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
                        if proc.returncode != 0:
                            out = proc.stdout or ''
                            low = out.lower()
                            if 'permission denied' in low or 'errno 13' in low or "could not install packages due to an oserror" in low:
                                LOGGER.warning("Auto-install of 'dill' failed due to permissions; continuing without dill. Output: %s", out)
                                return None
                            raise Exception(f"pip install returned {proc.returncode}: {proc.stdout}")
                        import importlib
                        importlib.invalidate_caches()
                        import dill as _dill
                        f.seek(0)
                        return _dill.load(f)
                    except Exception as ie2:
                        raise CustomException(
                            f"Failed to auto-install 'dill' while deserializing {path}.\nTried: {' '.join(cmd)}\nError: {ie2}\nPlease install dill manually: pip install dill",
                            sys,
                        ) from ie2
                raise CustomException(
                    f"Failed to deserialize {path}: unknown error and 'dill' is not installed.\n"
                    f"Try installing dill: pip install dill\nOriginal error: {e}",
                    sys,
                )
            except Exception as e2:
                raise CustomException(e2, sys)


def _strip_percent(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith('%'):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        import re
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            return float(m.group(0))
        return None


def reason_from_row(row: pd.Series) -> str:
    """Return a short human-readable reason(s) based on raw row fields."""
    reasons = []

    # numeric percent fields
    work = _strip_percent(row.get('Work Prog', np.nan))
    timep = _strip_percent(row.get('Time Prog', np.nan))
    effortp = _strip_percent(row.get('Effort Prog', np.nan))
    eac_eff = _strip_percent(row.get('EAC Efficiency', np.nan))

    try:
        if work is not None and timep is not None:
            if work + 20 < timep:
                reasons.append('Work progress significantly behind schedule (Work% << Time%)')
            elif work - 20 > timep:
                reasons.append('Work progress significantly ahead of schedule (Work% >> Time%)')
    except Exception:
        pass

    try:
        if effortp is not None:
            if effortp > 120:
                reasons.append('Effort consumed is much higher than planned (Effort Prog > 120%)')
            elif effortp < 50:
                reasons.append('Low effort consumption relative to plan (Effort Prog < 50%)')
    except Exception:
        pass

    try:
        if eac_eff is not None:
            if eac_eff < -5:
                reasons.append('EAC efficiency negative -> projected effort overrun')
            elif eac_eff < 20:
                reasons.append('Low EAC efficiency -> potential effort risk')
    except Exception:
        pass

    try:
        ps = row.get('Plan Start')
        act = row.get('Actual Start')
        if pd.notna(ps) and pd.notna(act):
            if str(act).strip() and str(ps).strip() and str(act).strip() != str(ps).strip():
                reasons.append('Actual start differs from planned start (possible late start)')
    except Exception:
        pass

    try:
        textual = row.get('RAG Reason + Observations')
        if pd.notna(textual) and isinstance(textual, str) and textual.strip():
            txt = textual.lower()
            if 'schedule overrun' in txt or 'overrun' in txt:
                reasons.append('Historic note: schedule overrun reported')
            if 'incomplete work' in txt or 'incomplete' in txt:
                reasons.append('Historic note: incomplete work reported')
            if 'effort' in txt and 'overrun' in txt:
                reasons.append('Historic note: effort overrun reported')
            if 'late start' in txt:
                reasons.append('Historic note: late start')
    except Exception:
        pass

    if not reasons:
        return ''
    unique = []
    for r in reasons:
        if r not in unique:
            unique.append(r)
    return '; '.join(unique)


def predict_with_reasons(input_csv: str = None, output_csv: str = None, auto_install_dill: bool = False):
    # If caller didn't provide an input, or provided path doesn't exist,
    # allow using an uploaded dataset placed in `uploads/data.csv` or a local
    # `data.csv` at repo root. This makes it easy to drop a CSV into the repo
    # and run predictions without changing the script.
    candidates = []
    if input_csv:
        candidates.append(input_csv)
    # common upload locations
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

    LOGGER.info(f'Using input CSV: {input_csv}')
    df = pd.read_csv(input_csv)

    # preserve project id
    id_col = None
    for c in ['DSF Project ID', 'Project ID', 'project_id']:
        if c in df.columns:
            id_col = c
            break

    # prepare features by dropping target and leakage columns
    blacklist = ['RAG', 'RAG Reason + Observations', 'Actual End', 'Closed Work']
    feature_df = df.drop(columns=[c for c in blacklist if c in df.columns], errors='ignore')

    raw_rows = feature_df.copy()

    if id_col and id_col in feature_df.columns:
        feature_df_for_transform = feature_df.drop(columns=[id_col])
    else:
        feature_df_for_transform = feature_df

    # --- Frequency encoding ---
    high_card_threshold = 20
    cat_cols_in_input = feature_df_for_transform.select_dtypes(include=['object', 'category']).columns.tolist()
    train_path = os.path.join('artifacts', 'train.csv')
    train_df = None
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        train_df.columns = train_df.columns.str.strip()
    for col in cat_cols_in_input:
        n_unique = feature_df_for_transform[col].nunique()
        if n_unique > high_card_threshold:
            freq = None
            if train_df is not None and col in train_df.columns:
                freq = train_df[col].value_counts(normalize=True)
            feature_df_for_transform[col] = feature_df_for_transform[col].map(freq).fillna(0) if freq is not None else 0

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
            LOGGER.warning(f"Preprocessor transform failed: {e}")
            transformed = None

    model_bundle = load_pickle(MODEL_BUNDLE_PATH, auto_install_dill=auto_install_dill)
    model = None
    label_encoder = None
    if model_bundle is not None and isinstance(model_bundle, dict) and 'model' in model_bundle:
        model = model_bundle.get('model')
        label_encoder = model_bundle.get('label_encoder')
    else:
        model = load_pickle(MODEL_PATH, auto_install_dill=auto_install_dill)
    if model is None:
        raise CustomException('No trained model found in artifacts. Run training first.', sys)

    # predict
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
        raise CustomException(e, sys)

    # --- Build full predictions_with_reasons.csv ---
    results = []
    for i in range(len(df)):
        pid = df[id_col].iloc[i] if id_col else i
        pred_label = preds[i] if preds is not None else None
        prob = float(pred_probs[i]) if pred_probs is not None and pred_probs[i] is not None else None
        
        actual_rag_reason = ''
        reason_parts = []

        # Get "RAG Reason" if available and not empty
        if 'RAG Reason' in df.columns:
            val = df['RAG Reason'].iloc[i]
        if pd.notna(val) and str(val).strip():
            reason_parts.append(str(val).strip())

        # Get "RAG Reason + Observations" if available and not empty
        if 'RAG Reason + Observations' in df.columns:
            val = df['RAG Reason + Observations'].iloc[i]
        if pd.notna(val) and str(val).strip():
            reason_parts.append(str(val).strip())

        # Join both parts into a single string
        actual_rag_reason = ' | '.join(reason_parts) if reason_parts else ''

        
        
        raw_row = raw_rows.iloc[i] if id_col is None else raw_rows.drop(columns=[id_col], errors='ignore').iloc[i]
        predicted_reason = reason_from_row(df.iloc[i])
        if not predicted_reason:
            feat_reasons = []
            try:
                if hasattr(model, 'feature_importances_') and transformed_feature_names is not None:
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

    # --- Compact predictions.csv (test rows only) ---
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
            LOGGER.warning(f'Failed to build compact predictions CSV: {e}')

    try:
        compact_df.to_csv(compact_path, index=False)
        LOGGER.info(f"Saved compact predictions to {compact_path} (rows={len(compact_df)})")
    except Exception as e:
        LOGGER.warning(f"Failed to write compact predictions CSV: {e}")
    else:
        print(f"Saved compact predictions to {compact_path}", flush=True)
        try:
            sys.stdout.flush()
        except Exception:
            pass

    # Also save a labeled confusion matrix derived from the compact predictions for quick comparison
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
            LOGGER.info(f"Saved confusion matrix derived from predictions to {cm_path}")
            print(f"Saved confusion matrix derived from predictions to {cm_path}", flush=True)
            try:
                sys.stdout.flush()
            except Exception:
                pass
    except Exception as e:
        LOGGER.warning(f"Failed to save confusion matrix from predictions: {e}")

    return out_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict RAG and produce reasons for DSF projects')
    parser.add_argument('--input', '-i', help='Input CSV path', default=DEFAULT_INPUT)
    parser.add_argument('--output', '-o', help='Output CSV path', default=DEFAULT_OUTPUT)
    parser.add_argument('--auto-install-dill', dest='auto_install_dill', action='store_true',
                        help='If set, attempt to pip install dill automatically when needed (requires network)')
    args = parser.parse_args()

    try:
        predict_with_reasons(input_csv=args.input, output_csv=args.output, auto_install_dill=getattr(args, 'auto_install_dill', False))
        # Signal a clear completion marker for supervising processes (dashboard)
        try:
            print('PREDICTION_COMPLETE', flush=True)
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as e:
        # If the failure relates to failing to auto-install dill due to permissions,
        # print a concise, user-friendly message instead of the full pip trace which
        # can pollute the Streamlit UI.
        msg = str(e)
        low = msg.lower()
        if 'failed to auto-install' in low and ('permission denied' in low or 'errno 13' in low or 'could not install packages due to an oserror' in low):
            print("Error during prediction: 'dill' is not installed and auto-install failed due to permissions.\nPlease install dill into the Python environment that runs the pipeline: `pip install dill`.")
            sys.exit(1)
        else:
            print(f"Error during prediction: {e}")
            raise
