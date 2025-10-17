import os
import sys
import pandas as pd
import numpy as np
import dill

ART = os.path.join('artifacts')
PRED = os.path.join(ART, 'predictions.csv')
CM_FILE = os.path.join(ART, 'confusion_matrix.csv')
BUNDLE = os.path.join(ART, 'model_bundle.pkl')
OUT = os.path.join(ART, 'compare_report.txt')

lines = []

# Load predictions
if not os.path.exists(PRED):
    print('predictions.csv not found at', PRED)
    sys.exit(1)
df = pd.read_csv(PRED)
if 'predicted' not in df.columns or 'actual' not in df.columns:
    print('predicted/actual columns missing in predictions.csv')
    sys.exit(1)

preds = df['predicted'].astype(str).str.strip()
actuals = df['actual'].astype(str).str.strip()
labels_union = sorted(list(set(actuals.tolist() + preds.tolist())))

lines.append(f'Found {len(df)} rows in predictions.csv')
lines.append('Labels (union): ' + ', '.join(labels_union))

# Confusion from predictions.csv
cm_from_preds = pd.crosstab(actuals, preds)
lines.append('\nConfusion matrix derived from predictions.csv (rows=actual, cols=predicted):')
lines.append(cm_from_preds.to_string())

# Load confusion_matrix.csv produced by training (may be numeric indexed)
if os.path.exists(CM_FILE):
    try:
        cm_train_df = pd.read_csv(CM_FILE, index_col=0)
        lines.append('\nLoaded confusion_matrix.csv from training artifacts:')
        lines.append(cm_train_df.to_string())

        # Try to map numeric indices to labels using model bundle
        mapped = False
        if os.path.exists(BUNDLE):
            try:
                with open(BUNDLE, 'rb') as f:
                    bundle = dill.load(f)
                le = bundle.get('label_encoder', None)
                if le is not None:
                    n = cm_train_df.shape[0]
                    try:
                        text_labels = list(le.inverse_transform(list(range(n))))
                        # rename rows/cols
                        cm_train_labeled = cm_train_df.copy()
                        cm_train_labeled.index = text_labels
                        cm_train_labeled.columns = text_labels
                        lines.append('\nMapped training confusion matrix to textual labels using saved label encoder:')
                        lines.append(cm_train_labeled.to_string())
                        mapped = True
                    except Exception as ex:
                        lines.append('\nFailed to inverse_transform label encoder: ' + str(ex))
            except Exception as ex:
                lines.append('\nFailed to load model_bundle.pkl: ' + str(ex))
        if not mapped:
            # if columns are numeric strings but actually textual labels, keep as is
            try:
                # attempt to interpret header values as labels
                cols = list(cm_train_df.columns.astype(str))
                if set(cols).issubset(set(labels_union)):
                    cm_train_labeled = cm_train_df.copy()
                else:
                    # fallback: keep numeric indices but prefix with 'idx:'
                    newlabels = ['idx:'+str(x) for x in cm_train_df.index.astype(str)]
                    cm_train_labeled = cm_train_df.copy()
                    cm_train_labeled.index = newlabels
                    cm_train_labeled.columns = newlabels
                lines.append('\nTraining confusion matrix (interpreted):')
                lines.append(cm_train_labeled.to_string())
            except Exception as ex:
                lines.append('\nFailed to interpret confusion_matrix.csv: ' + str(ex))
    except Exception as e:
        lines.append('\nFailed to read confusion_matrix.csv: ' + str(e))
else:
    lines.append('\nconfusion_matrix.csv not found in artifacts')

# Compare totals
try:
    # sum of diagonal from predictions-derived
    tp_preds = cm_from_preds.values.diagonal().sum() if not cm_from_preds.empty else 0
    total_preds = cm_from_preds.values.sum() if not cm_from_preds.empty else len(df)
    lines.append(f'\nPredictions-derived: total samples={total_preds}, total correct (diag sum)={tp_preds}')
except Exception:
    pass

# Save report
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Wrote comparison report to', OUT)
print('\n'.join(lines))
