from src.pipeline.predict_pipeline import predict_with_reasons
print('module loaded')
try:
    r = predict_with_reasons(input_csv='notebook/data/Project_Progress_Report_Status.csv', output_csv='artifacts/predictions_with_reasons_smoketest.csv', auto_install_dill=False)
    print('smoke run done, rows=', len(r))
except Exception as e:
    print('smoke run failed:', e)
