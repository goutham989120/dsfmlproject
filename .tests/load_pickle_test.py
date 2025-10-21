from src.pipeline.predict_pipeline import load_pickle
print('callable?', callable(load_pickle))
print('load existing preprocessor:', load_pickle('artifacts/preprocessor.pkl'))
