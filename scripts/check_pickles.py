import os
import sys
import pickle
print('CWD:', os.getcwd())
files = ['artifacts/model_bundle.pkl', 'artifacts/model.pkl', 'artifacts/preprocessor.pkl']
for f in files:
    print('\n---', f)
    print('exists:', os.path.exists(f))
    if not os.path.exists(f):
        continue
    try:
        print('size:', os.path.getsize(f))
    except Exception as e:
        print('size error:', e)
    try:
        with open(f, 'rb') as fh:
            try:
                obj = pickle.load(fh)
                print('pickle.load OK; type=', type(obj))
                # If it's a dict with model, show keys
                try:
                    if isinstance(obj, dict):
                        print('dict keys:', list(obj.keys()))
                except Exception:
                    pass
            except Exception as e:
                print('pickle.load failed:', repr(e))
    except Exception as e:
        print('open failed:', repr(e))
    try:
        import dill
        with open(f, 'rb') as fh:
            try:
                fh.seek(0)
                obj = dill.load(fh)
                print('dill.load OK; type=', type(obj))
            except Exception as e:
                print('dill.load failed:', repr(e))
    except Exception as e:
        print('dill not available:', repr(e))

print('\nPython executable:', sys.executable)
print('Python version:', sys.version)
