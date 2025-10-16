DSFML Project Dashboard

This simple Streamlit dashboard provides a quick, sharable summary of the model predictions to present to hackathon judges.

How to run

1. Ensure the training + prediction pipeline has run and produced `artifacts/predictions_with_reasons.csv`.
2. Create a virtual environment and install dependencies (from workspace root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r dashboard\requirements.txt
```

3. Run the dashboard:

```powershell
streamlit run dashboard\app.py
```

What it shows
- Dataset summary and column list
- Predicted RAG distribution and probability histogram
- Predicted vs Actual cross-tab
- Top predicted reasons and a sample of projects
- Download button to export filtered CSV

Notes
- `predicted_EAC_date` is a best-effort field derived from `Replan End` -> `Plan End` -> `Actual End`.
- If you want more visuals (confusion matrix heatmap, time-series of predicted EAC), tell me and I will add them.
