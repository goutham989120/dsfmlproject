import streamlit as st
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess
from pathlib import Path

st.set_page_config(page_title='DSFML Project — Predictions Dashboard', layout='wide')

st.title('DSFML Project — Predictions Dashboard')


def safe_rerun():
    """Attempt to rerun the Streamlit script in a backward-compatible way."""
    try:
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
            return
    except Exception:
        pass
    try:
        import time
        if hasattr(st, 'experimental_set_query_params'):
            st.experimental_set_query_params(_refresh=int(time.time()))
    except Exception:
        pass
    # fallback
    try:
        st.stop()
    except Exception:
        return

# Dataset scope selector: All predictions vs test-only predictions
scope = st.selectbox('Dataset scope', options=['All predictions', 'Test rows only'], index=0, help='Choose full predictions_with_reasons (All) or compact predictions.csv (Test rows only)')

@st.cache_data
def load_full():
    path = os.path.join('artifacts', 'predictions_with_reasons.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_compact():
    path = os.path.join('artifacts', 'predictions.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

if scope == 'All predictions':
    df = load_full()
else:
    # compact predictions.csv has columns 'predicted' and 'actual' - normalize names
    tmp = load_compact()
    if not tmp.empty:
        # try to map columns to expected names used downstream
        col_map = {}
        if 'predicted' in tmp.columns:
            col_map['predicted'] = 'predicted_RAG'
        if 'actual' in tmp.columns:
            col_map['actual'] = 'actual_RAG'
        # copy other helpful columns
        if 'predicted_RAG_reason' in tmp.columns:
            col_map['predicted_RAG_reason'] = 'predicted_RAG_reason'
        if 'actual_RAG_reason' in tmp.columns:
            col_map['actual_RAG_reason'] = 'actual_RAG_reason'
        if 'predicted_probability' in tmp.columns:
            col_map['predicted_probability'] = 'predicted_probability'
        if 'predicted_EAC_date' in tmp.columns:
            col_map['predicted_EAC_date'] = 'predicted_EAC_date'
        df = tmp.rename(columns=col_map).copy()

if df.empty:
    st.error('No predictions found for selected scope. Run the pipeline or pick a different scope.')
    st.stop()

# Try to compute basic metrics if actual_RAG present
metrics = {}
if 'actual_RAG' in df.columns and 'predicted_RAG' in df.columns:
    try:
        y_true = df['actual_RAG'].astype(str)
        y_pred = df['predicted_RAG'].astype(str)
        acc = accuracy_score(y_true, y_pred)
        rep = classification_report(y_true, y_pred, output_dict=True)
        metrics['accuracy'] = acc
        metrics['report'] = rep
    except Exception:
        metrics = {}

# Model insights (optional)
model_bundle = None
feature_importances = None
model_features = None
bundle_path = os.path.join('artifacts', 'model_bundle.pkl')
try:
    if os.path.exists(bundle_path):
        with open(bundle_path, 'rb') as f:
            model_bundle = pickle.load(f)
        model = model_bundle.get('model') if isinstance(model_bundle, dict) else None
        if model is not None and hasattr(model, 'feature_importances_'):
            fi = np.array(model.feature_importances_)
            # try to get feature names from preprocessor
            try:
                preproc_path = os.path.join('artifacts', 'preprocessor.pkl')
                if os.path.exists(preproc_path):
                    with open(preproc_path, 'rb') as pf:
                        pre = pickle.load(pf)
                    try:
                        model_features = list(pre.get_feature_names_out())
                    except Exception:
                        # fallback: None
                        model_features = None
            except Exception:
                model_features = None
            if model_features is not None and len(model_features) == fi.shape[0]:
                feature_importances = pd.Series(fi, index=model_features).sort_values(ascending=False)
            else:
                feature_importances = pd.Series(fi).sort_values(ascending=False)
except Exception:
    model_bundle = None
    feature_importances = None
# Basic summary
with st.expander('Dataset summary'):
    st.write('Rows:', len(df))
    st.write('Columns:', list(df.columns))
    st.write(df.describe(include='all'))

# Layout: left filters, right main visualisations
left, right = st.columns([1, 3])

with left:
    st.header('Filters')
    # Predict button
    st.markdown('### Run prediction')
    st.write('Run the prediction pipeline against the raw input and regenerate visuals for demo.')
    # File uploader: if user uploads a CSV, save it to uploads/data.csv and use it as input
    uploaded_path = None
    uploaded_file = st.file_uploader('Upload CSV for prediction (optional)', type=['csv'])
    if uploaded_file is not None:
        try:
            uploads_dir = Path('uploads')
            uploads_dir.mkdir(parents=True, exist_ok=True)
            # Save uploaded file to disk so pipeline (which runs as subprocess) can access it
            target = uploads_dir / 'data.csv'
            # If a file already exists, allow overwrite
            with open(target, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success(f'Saved uploaded file to {str(target)}')
            uploaded_path = str(target)
        except Exception as ex:
            st.error(f'Failed to save uploaded file: {ex}')
    if st.button('Predict now'):
        with st.spinner('Running prediction pipeline...'):
            root = Path.cwd()
            # call the predict pipeline script using the Python executable
            try:
                predict_script = str(root / 'src' / 'pipeline' / 'predict_pipeline.py')
                cmd = [sys.executable, predict_script]
                if uploaded_path:
                    cmd += ['--input', uploaded_path]

                # Run process and stream stdout/stderr to the UI so users see progress.
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                placeholder = st.empty()
                lines = []
                import time
                start = time.time()
                timeout_seconds = 600  # 10 minutes max by default
                killed = False
                while True:
                    if proc.stdout is None:
                        break
                    line = proc.stdout.readline()
                    if line:
                        lines.append(line)
                        # Keep the displayed output reasonably sized
                        display_text = ''.join(lines[-1000:])
                        placeholder.text(display_text)
                    if proc.poll() is not None:
                        # process finished
                        # read remaining
                        remainder = proc.stdout.read()
                        if remainder:
                            lines.append(remainder)
                            placeholder.text(''.join(lines[-1000:]))
                        break
                    # timeout guard
                    if (time.time() - start) > timeout_seconds:
                        try:
                            proc.kill()
                            killed = True
                        except Exception:
                            pass
                        lines.append('\n[Process killed after timeout]')
                        placeholder.text(''.join(lines[-1000:]))
                        break
                    time.sleep(0.1)

                retcode = proc.poll()
                st.write('Prediction exit code:', retcode)
                if killed:
                    st.warning('Prediction process was killed due to timeout.')
            except Exception as ex:
                st.error('Failed to run prediction pipeline: ' + str(ex))
            # regenerate visuals: run helper scripts and stream their output with timeouts
            scripts = [root / 'gen_pred_confusion_matrix.py', root / 'compare_preds.py', root / 'analyze_mismatches.py']
            post_placeholder = st.empty()
            for s in scripts:
                if not s.exists():
                    continue
                try:
                    cmd = [sys.executable, str(s)]
                    post_placeholder.markdown(f'**Running:** `{s.name}`')
                    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    out_lines = []
                    import time
                    start2 = time.time()
                    # shorter timeout for post scripts
                    timeout2 = 120
                    while True:
                        if proc2.stdout is None:
                            break
                        line2 = proc2.stdout.readline()
                        if line2:
                            out_lines.append(line2)
                            post_placeholder.text(''.join(out_lines[-500:]))
                        if proc2.poll() is not None:
                            rem = proc2.stdout.read()
                            if rem:
                                out_lines.append(rem)
                                post_placeholder.text(''.join(out_lines[-500:]))
                            break
                        if (time.time() - start2) > timeout2:
                            try:
                                proc2.kill()
                                out_lines.append('\n[Script killed after timeout]')
                                post_placeholder.text(''.join(out_lines[-500:]))
                            except Exception:
                                pass
                            break
                        time.sleep(0.05)
                    ret2 = proc2.poll()
                    post_placeholder.markdown(f'`{s.name}` exit code: {ret2}')
                except Exception as ex:
                    st.warning(f'Failed to run {s.name}: {ex}')
            # clear cache and rerun (safe)
            try:
                st.cache_data.clear()
            except Exception:
                pass
            # Use experimental_rerun if available, else fallback to setting a query param and stop
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    import time
                    st.experimental_set_query_params(_refresh=int(time.time()))
                except Exception:
                    pass
                st.stop()
    rag_vals = df['predicted_RAG'].dropna().unique().tolist() if 'predicted_RAG' in df.columns else []
    sel_rag = st.multiselect('Predicted RAG', options=rag_vals, default=rag_vals)

    if 'actual_RAG' in df.columns:
        actual_vals = df['actual_RAG'].dropna().unique().tolist()
        sel_actual = st.multiselect('Actual RAG', options=actual_vals, default=actual_vals)
    else:
        sel_actual = None

    min_prob = float(df['predicted_probability'].min()) if 'predicted_probability' in df.columns else 0.0
    max_prob = float(df['predicted_probability'].max()) if 'predicted_probability' in df.columns else 1.0
    prob_thresh = st.slider('Probability threshold', min_value=0.0, max_value=1.0, value=(min_prob, max_prob))

# Apply filters
q = df.copy()
if sel_rag:
    q = q[q['predicted_RAG'].isin(sel_rag)]
if sel_actual is not None:
    q = q[q['actual_RAG'].isin(sel_actual)]
if 'predicted_probability' in q.columns:
    q = q[(q['predicted_probability'] >= prob_thresh[0]) & (q['predicted_probability'] <= prob_thresh[1])]

with right:
    st.header('Overview')
    # Top KPIs
    k1, k2, k3 = st.columns(3)
    if 'accuracy' in metrics:
        k1.metric('Accuracy', f"{metrics['accuracy']:.3f}")
    else:
        k1.metric('Accuracy', 'N/A')
    k2.metric('Rows', len(df))
    if 'predicted_probability' in df.columns:
        k3.metric('Avg predicted prob', f"{df['predicted_probability'].mean():.2f}")
    else:
        k3.metric('Avg predicted prob', 'N/A')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Predicted RAG distribution')
        fig, ax = plt.subplots()
        sns.countplot(data=q, x='predicted_RAG', order=sorted(q['predicted_RAG'].unique()), ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader('Probability histogram')
        fig2, ax2 = plt.subplots()
        if 'predicted_probability' in q.columns and q['predicted_probability'].dropna().shape[0] > 0:
            sns.histplot(q['predicted_probability'].dropna(), bins=20, kde=True, ax=ax2)
            st.pyplot(fig2)
        else:
            st.write('No predicted probability available for the selected dataset/scope.')

    if 'actual_RAG' in q.columns:
        st.subheader('Predicted vs Actual')
        ct = pd.crosstab(q['actual_RAG'], q['predicted_RAG'], normalize='index')
        st.dataframe(ct)
        # confusion matrix heatmap
        try:
            y_true = q['actual_RAG'].astype(str)
            y_pred = q['predicted_RAG'].astype(str)
            labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
        except Exception:
            pass

    st.subheader('Top predicted projects (sample)')
    show_cols = ['DSF Project ID', 'predicted_RAG', 'predicted_probability', 'predicted_EAC_date', 'actual_RAG', 'actual_RAG_reason', 'predicted_RAG_reason']
    show_cols = [c for c in show_cols if c in q.columns]
    st.dataframe(q[show_cols].head(50))

    # Top reasons
    st.subheader('Top predicted reasons (text)')
    reasons = q['predicted_RAG_reason'].dropna().astype(str)
    top_reasons = reasons.value_counts().head(20)
    st.bar_chart(top_reasons)

    # Feature importances
    if feature_importances is not None:
        st.subheader('Model feature importances')
        st.write('Top features influencing the model (if available from saved model)')
        st.dataframe(feature_importances.head(20))
        # downloadable image of top features
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        feature_importances.head(20).sort_values().plot(kind='barh', ax=ax_fi)
        ax_fi.set_ylabel('Feature')
        ax_fi.set_xlabel('Importance')
        st.pyplot(fig_fi)
        buf = BytesIO()
        fig_fi.savefig(buf, format='png')
        buf.seek(0)
        st.download_button('Download feature importance PNG', data=buf, file_name='feature_importances.png', mime='image/png')

# Download button
st.markdown('---')
@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

csv_bytes = to_csv_bytes(q)
st.download_button(label='Download filtered CSV', data=csv_bytes, file_name='predictions_filtered.csv', mime='text/csv')

st.markdown('### Notes')
st.write('- predicted_EAC_date is a best-effort field selected from Replan End -> Plan End -> Actual End in the input CSV.')
st.write('- actual_RAG and actual_RAG_reason are included when available.')
st.write('- If you want a presentation-friendly PDF or slides, I can add an export that snapshots key charts into a single file.')


