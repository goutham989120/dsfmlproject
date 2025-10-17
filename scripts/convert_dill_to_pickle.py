"""
Convert artifacts serialized with dill into stdlib pickle where possible.

Behavior:
- Scans artifacts/ for files ending with .pkl
- For each file:
  - Try to load with stdlib pickle. If success, skip.
  - Else try to load with dill. If success, back up original to .dillbak and re-dump using pickle.HIGHEST_PROTOCOL.
  - If unable to load with either, log and skip.

Note: This script requires `dill` to be installed in the environment when dill-serialized files exist.
"""
import os
import glob
import sys
import shutil
import pickle

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')
if not os.path.isdir(ARTIFACT_DIR):
    ARTIFACT_DIR = os.path.join(os.getcwd(), 'artifacts')

pkl_files = glob.glob(os.path.join(ARTIFACT_DIR, '*.pkl'))
if not pkl_files:
    print('No .pkl files found in', ARTIFACT_DIR)
    sys.exit(0)

# Helper to attempt pickle.load
def try_pickle_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return e

# Helper to attempt dill.load
def try_dill_load(path):
    try:
        import dill
    except Exception as e:
        return e
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        return e

converted = []
skipped = []
failed = []
for p in pkl_files:
    print('\nProcessing', p)
    res = try_pickle_load(p)
    if not isinstance(res, Exception):
        print(' - Already loadable with stdlib pickle; skipping')
        skipped.append(p)
        continue

    print(' - Not loadable with stdlib pickle; attempting dill...')
    res2 = try_dill_load(p)
    if isinstance(res2, Exception):
        print(' - Failed to load with dill:', res2)
        failed.append((p, res2))
        continue

    obj = res2
    backup = p + '.dillbak'
    try:
        shutil.copy2(p, backup)
        print(f' - Backed up original to {backup}')
    except Exception as e:
        print(' - Warning: failed to backup original file:', e)

    try:
        with open(p, 'wb') as out:
            pickle.dump(obj, out, protocol=pickle.HIGHEST_PROTOCOL)
        print(' - Re-saved with stdlib pickle (protocol=' + str(pickle.HIGHEST_PROTOCOL) + ')')
        converted.append(p)
    except Exception as e:
        print(' - Failed to re-save with stdlib pickle:', e)
        failed.append((p, e))

print('\nSummary:')
print(' Converted:', len(converted))
for c in converted:
    print('  -', c)
print(' Skipped (already pickle-loadable):', len(skipped))
for s in skipped:
    print('  -', s)
print(' Failed to convert:', len(failed))
for fpath, err in failed:
    print('  -', fpath, '->', err)

if failed:
    print('\nSome files could not be converted automatically. Consider installing dill manually and inspecting those files.')
    sys.exit(2)

sys.exit(0)
