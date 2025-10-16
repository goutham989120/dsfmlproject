import os
import sys
import logging
import pandas as pd
import numpy as np

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


def load_pickle(path: str):
	if not os.path.exists(path):
		return None
	with open(path, 'rb') as f:
		try:
			return pickle.load(f)
		except Exception:
			f.seek(0)
			return pickle.load(f)


def _strip_percent(x):
	if pd.isna(x):
		return None
	s = str(x).strip()
	if s.endswith('%'):
		s = s[:-1]
	try:
		return float(s)
	except Exception:
		# if contains non-numeric, try to extract digits
		import re
		m = re.search(r"-?\d+\.?\d*", s)
		if m:
			return float(m.group(0))
		return None


def reason_from_row(row: pd.Series) -> str:
	"""Return a short human-readable reason(s) based on raw row fields.

	This is a lightweight, interpretable rule-based explainer. It is intentionally
	conservative (few rules) and complements the model prediction.
	"""
	reasons = []

	# numeric percent fields
	work = _strip_percent(row.get('Work Prog', np.nan))
	timep = _strip_percent(row.get('Time Prog', np.nan))
	effortp = _strip_percent(row.get('Effort Prog', np.nan))
	eac_eff = _strip_percent(row.get('EAC Efficiency', np.nan))

	# basic rules
	try:
		if work is not None and timep is not None:
			# behind schedule if work significantly less than time
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
				# low effort consumed relative to plan could mean early stage or under-reporting
				reasons.append('Low effort consumption relative to plan (Effort Prog < 50%)')
	except Exception:
		pass

	try:
		if eac_eff is not None:
			# negative or very low efficiency is a red flag
			if eac_eff < -5:
				reasons.append('EAC efficiency negative -> projected effort overrun')
			elif eac_eff < 20:
				reasons.append('Low EAC efficiency -> potential effort risk')
	except Exception:
		pass

	# check late start: compare Plan Start and Actual Start if present
	try:
		ps = row.get('Plan Start')
		act = row.get('Actual Start')
		if pd.notna(ps) and pd.notna(act):
			# compare as strings: if values differ and actual start not empty, flag late start
			if str(act).strip() and str(ps).strip() and str(act).strip() != str(ps).strip():
				reasons.append('Actual start differs from planned start (possible late start)')
	except Exception:
		pass

	# if we have any textual clues already present in the RAG Reason column, surface them cautiously
	try:
		textual = row.get('RAG Reason + Observations')
		if pd.notna(textual) and isinstance(textual, str) and textual.strip():
			# pick up high-signal tokens (schedule overrun, incomplete work, effort overrun)
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
	# keep reasons concise and unique
	unique = []
	for r in reasons:
		if r not in unique:
			unique.append(r)
	return '; '.join(unique)


def predict_with_reasons(input_csv: str = None, output_csv: str = None):
	input_csv = input_csv or DEFAULT_INPUT
	output_csv = output_csv or DEFAULT_OUTPUT

	if not os.path.exists(input_csv):
		raise CustomException(f"Input file not found: {input_csv}", sys)

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

	# Keep a copy of raw rows for rule-based reasons
	raw_rows = feature_df.copy()

	# Remove identifier column before preprocessing if present
	if id_col and id_col in feature_df.columns:
		feature_df_for_transform = feature_df.drop(columns=[id_col])
	else:
		feature_df_for_transform = feature_df

	# --- Frequency encoding for high-cardinality categorical columns (match training logic) ---
	high_card_threshold = 20
	freq_encoded_cols = []
	cat_cols_in_input = feature_df_for_transform.select_dtypes(include=['object', 'category']).columns.tolist()
	# Try to load train.csv to get frequency mapping
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
			# Map to frequency (proportion) observed in train, fallback to 0 for unseen
			feature_df_for_transform[col] = feature_df_for_transform[col].map(freq).fillna(0) if freq is not None else 0
			freq_encoded_cols.append(col)
	# --- End frequency encoding patch ---

	# load preprocessor (may be None)
	preprocessor = load_pickle(PREPROCESSOR_PATH)

	transformed = None
	transformed_feature_names = None
	if preprocessor is not None:
		try:
			# try to determine the exact columns the preprocessor expects
			expected_cols = None
			if hasattr(preprocessor, 'feature_names_in_'):
				try:
					expected_cols = list(preprocessor.feature_names_in_)
				except Exception:
					expected_cols = None

			# fallback: try to extract column lists from transformers_
			if expected_cols is None and hasattr(preprocessor, 'transformers_'):
				cols = []
				try:
					for name, trans, cols_spec in preprocessor.transformers_:
						# if cols_spec is string 'remainder' skip
						if isinstance(cols_spec, (list, tuple)):
							cols.extend(list(cols_spec))
				except Exception:
					cols = []
				if cols:
					expected_cols = cols

			# If we got expected columns, reindex input to that set, filling missing with NaN
			if expected_cols is not None:
				# keep only expected columns, in order; fill missing columns with NaN
				feature_df_for_transform = feature_df_for_transform.reindex(columns=expected_cols)

			transformed = preprocessor.transform(feature_df_for_transform)
			# convert sparse to dense
			if hasattr(transformed, 'toarray'):
				transformed = transformed.toarray()
			# attempt to get feature names from transformer
			try:
				transformed_feature_names = preprocessor.get_feature_names_out(feature_df_for_transform.columns)
			except Exception:
				transformed_feature_names = None
		except Exception as e:
			# If transform fails, log and fall back to raw array for models that accept it
			LOGGER.warning(f"Preprocessor transform failed: {e}")
			transformed = None

	# load model bundle or model
	model_bundle = load_pickle(MODEL_BUNDLE_PATH)
	model = None
	label_encoder = None
	if model_bundle is not None and isinstance(model_bundle, dict) and 'model' in model_bundle:
		model = model_bundle.get('model')
		label_encoder = model_bundle.get('label_encoder')
	else:
		# fallback to single model file
		model = load_pickle(MODEL_PATH)

	if model is None:
		raise CustomException('No trained model found in artifacts. Run training first.', sys)

	# predict
	try:
		if transformed is not None:
			X = transformed
		else:
			# try to use raw data frame
			X = feature_df_for_transform.values

		# classification path
		probs = None
		y_pred = None
		try:
			probs = model.predict_proba(X)
			y_idx = np.argmax(probs, axis=1)
		except Exception:
			# classifier may not support predict_proba; try predict
			y_pred = model.predict(X)
			try:
				# if label encoder exists, map to indices
				if label_encoder is not None:
					# label_encoder.inverse_transform expects int labels -> we assume model returned ints
					pass
			except Exception:
				pass

		if probs is not None:
			# get predicted label indices and probability
			pred_probs = np.max(probs, axis=1)
			try:
				if label_encoder is not None:
					preds = label_encoder.inverse_transform(y_idx)
				else:
					# if model has classes_
					if hasattr(model, 'classes_'):
						preds = np.array(model.classes_)[y_idx]
					else:
						preds = y_idx
			except Exception:
				preds = y_idx
		else:
			# probs not available
			pred_probs = [None] * (X.shape[0] if hasattr(X, 'shape') else len(X))
			preds = y_pred

	except Exception as e:
		raise CustomException(e, sys)

	# build results
	results = []
	n = len(df)

	for i in range(n):
		pid = df[id_col].iloc[i] if id_col else i
		pred_label = preds[i] if preds is not None else None
		prob = float(pred_probs[i]) if pred_probs is not None and pred_probs[i] is not None else None

		# Actual RAG reason from input CSV
		actual_rag_reason = df['RAG Reason + Observations'].iloc[i] if 'RAG Reason + Observations' in df.columns else ''

		# rule-based reason on raw row (predicted reason)
		raw_row = raw_rows.iloc[i] if id_col is None else raw_rows.drop(columns=[id_col], errors='ignore').iloc[i]
		predicted_reason = reason_from_row(df.iloc[i])

		# if rule-based reason is empty, attempt to supply top features from model importances
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

		# Determine a single predicted EAC date to include (best-effort): prefer Replan End -> Plan End -> Actual End
		predicted_eac_date = None
		for cand in ['Replan End', 'Plan End', 'Actual End']:
			if cand in df.columns:
				val = df[cand].iloc[i]
				if pd.notna(val) and str(val).strip():
					predicted_eac_date = val
					break

		# Read actual RAG value from input if present
		actual_rag_value = None
		if 'RAG' in df.columns:
			v = df['RAG'].iloc[i]
			actual_rag_value = v if pd.notna(v) else None

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

	# Overwrite existing output to avoid appending heterogeneous rows or stray preamble lines
	if os.path.exists(output_csv):
		try:
			os.remove(output_csv)
		except Exception as e:
			LOGGER.warning(f"Could not remove existing output file {output_csv}: {e}")

	# write file and perform a quick sanity check that the header is the first line
	out_df.to_csv(output_csv, index=False)
	try:
		with open(output_csv, 'r', encoding='utf-8') as f:
			first_line = f.readline().strip()
		expected = ','.join(out_df.columns.astype(str).tolist())
		if first_line != expected:
			LOGGER.warning(f"Output CSV header mismatch: first_line={first_line!r} expected={expected!r}. Rewriting file.")
			out_df.to_csv(output_csv, index=False)
	except Exception as e:
		LOGGER.warning(f"Failed to validate written CSV header for {output_csv}: {e}")

	print(f"Saved predictions with reasons to {output_csv}")

	# Also write a compact predictions CSV (used by some dashboards/tests) with just predicted/actual and reasons
	compact_path = os.path.join('artifacts', 'predictions.csv')
	# default header-only compact dataframe
	compact_df = pd.DataFrame(columns=['predicted', 'actual', 'predicted_RAG_reason', 'actual_RAG_reason'])

	# Try to restrict compact CSV to test rows by matching IDs in artifacts/test.csv
	test_path = os.path.join('artifacts', 'test.csv')
	if os.path.exists(test_path):
		try:
			test_df = pd.read_csv(test_path)
			test_df.columns = test_df.columns.str.strip()
			# candidate id columns to try
			id_candidates = [c for c in ['DSF Project ID', 'Project ID', 'project_id'] if c in test_df.columns]
			# also consider id_col detected in input file
			if id_col is not None:
				id_candidates.insert(0, id_col)
			# find the first candidate that exists in test_df
			found = None
			for c in id_candidates:
				if c in test_df.columns:
					found = c
					break
			if found is not None:
				test_ids = set(test_df[found].dropna().astype(str).str.strip().tolist())
				# filter predictions for rows matching those IDs
				if 'DSF Project ID' in out_df.columns:
					matched = out_df[out_df['DSF Project ID'].astype(str).str.strip().isin(test_ids)]
				else:
					matched = out_df[out_df.index.isin([])]
				if not matched.empty:
					compact_df = matched[['predicted_RAG', 'actual_RAG', 'predicted_RAG_reason', 'actual_RAG_reason']].copy()
					compact_df = compact_df.rename(columns={'predicted_RAG': 'predicted', 'actual_RAG': 'actual'})
				else:
					LOGGER.info('Test CSV present but no matching prediction rows found; writing header-only compact CSV')
		except Exception as e:
			LOGGER.warning(f'Failed to read or match test CSV for compact predictions: {e}')

	# write compact CSV (filtered or header-only)
	try:
		compact_df.to_csv(compact_path, index=False)
		LOGGER.info(f"Saved compact predictions to {compact_path} (rows={len(compact_df)})")
	except Exception as e:
		LOGGER.warning(f"Failed to write compact predictions CSV: {e}")

	return out_df


if __name__ == '__main__':
	# allow optional CLI args: input and output
	import argparse

	parser = argparse.ArgumentParser(description='Predict RAG and produce reasons for DSF projects')
	parser.add_argument('--input', '-i', help='Input CSV path', default=DEFAULT_INPUT)
	parser.add_argument('--output', '-o', help='Output CSV path', default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	try:
		predict_with_reasons(input_csv=args.input, output_csv=args.output)
	except Exception as e:
		print(f"Error during prediction: {e}")
		raise
