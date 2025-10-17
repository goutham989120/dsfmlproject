import pandas as pd
import os

ART='artifacts'
PRED_FULL=os.path.join(ART,'predictions_with_reasons.csv')
TEST=os.path.join(ART,'test.csv')
OUT=os.path.join(ART,'mismatch_details.txt')

if not os.path.exists(PRED_FULL) or not os.path.exists(TEST):
    print('Required files missing:', PRED_FULL, TEST)
    raise SystemExit(1)

pred = pd.read_csv(PRED_FULL)
test = pd.read_csv(TEST)

# normalize id column
id_cols = [c for c in ['DSF Project ID','Project ID','project_id'] if c in test.columns]
if id_cols:
    id_col = id_cols[0]
else:
    # fallback to index
    test['DSF Project ID'] = test.index.astype(str)
    id_col = 'DSF Project ID'

pred['DSF Project ID'] = pred['DSF Project ID'].astype(str).str.strip()
test[id_col] = test[id_col].astype(str).str.strip()

# join
merged = pd.merge(pred, test, left_on='DSF Project ID', right_on=id_col, how='inner', suffixes=('_pred','_test'))
merged['predicted_RAG'] = merged['predicted_RAG'].astype(str).str.strip()
if 'RAG' in merged.columns:
    merged['actual_RAG_test'] = merged['RAG'].astype(str).str.strip()
else:
    merged['actual_RAG_test'] = merged['actual_RAG'] if 'actual_RAG' in merged.columns else None

# correctness
merged['correct'] = merged['predicted_RAG'] == merged['actual_RAG_test']

# summary
total = len(merged)
correct = merged['correct'].sum()

lines=[]
lines.append(f'Merged rows: {total}, correct: {correct}, accuracy: {correct/total:.4f}')

# per-class breakdown
for label in sorted(list(set(merged['actual_RAG_test'].dropna().unique()))):
    sub = merged[merged['actual_RAG_test']==label]
    tot = len(sub)
    corr = (sub['correct']).sum()
    acc = (corr/tot) if tot>0 else 0.0
    lines.append(f"Actual {label}: total={tot}, correct={corr}, accuracy={acc:.4f}")
    # list misclassified ids (limit 10)
    mis = sub[~sub['correct']]
    if not mis.empty:
        # after merge the original prediction id column is 'DSF Project ID' (from left)
        id_col_pred = 'DSF Project ID'
        if id_col_pred in mis.columns:
            ids = mis[id_col_pred].astype(str).tolist()[:20]
        else:
            # fallback to any available id-like column
            possible = [c for c in mis.columns if 'id' in c.lower() or 'project' in c.lower()]
            ids = mis[possible[0]].astype(str).tolist()[:20] if possible else []
        lines.append(' Misclassified IDs (sample up to 20): ' + ', '.join(ids))

with open(OUT,'w',encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('\n'.join(lines))
print('Wrote details to', OUT)
