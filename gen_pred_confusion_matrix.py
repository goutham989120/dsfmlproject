import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ART = os.path.join('artifacts')
PRED = os.path.join(ART, 'predictions.csv')
OUT_PNG = os.path.join(ART, 'Pred_confusion_matrix.png')

if not os.path.exists(PRED):
    print('predictions.csv not found at', PRED)
    sys.exit(1)

df = pd.read_csv(PRED)
# try common column names
pred_col = None
actual_col = None
for c in ['predicted', 'prediction', 'pred']:
    if c in df.columns:
        pred_col = c
        break
for c in ['actual', 'label', 'target']:
    if c in df.columns:
        actual_col = c
        break
if pred_col is None or actual_col is None:
    print('Could not find predicted/actual columns in', PRED)
    print('Columns found:', df.columns.tolist())
    sys.exit(1)

preds = df[pred_col].astype(str).str.strip()
actuals = df[actual_col].astype(str).str.strip()
labels = sorted(list(set(actuals.tolist() + preds.tolist())))

# compute confusion matrix
cm = confusion_matrix(actuals, preds, labels=labels)

# plot
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=False)
# annotate counts on cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.title('Confusion matrix (derived from predictions.csv)')
plt.tight_layout()

# ensure artifacts dir exists
os.makedirs(ART, exist_ok=True)
plt.savefig(OUT_PNG, dpi=150)
print('Wrote', OUT_PNG)
