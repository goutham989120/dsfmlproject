from pathlib import Path
import runpy
import sys

# This is a tiny wrapper so users can run `streamlit run dashboard/app.py`
# even when their current working directory is `notebook/data`.
# It locates the real dashboard under the repository root and executes it.

HERE = Path(__file__).resolve()
# wrapper location is expected to be: <repo>/notebook/data/dashboard/app.py
# repo root = parents[3]
repo_root = HERE.parents[3]
real = repo_root / 'dashboard' / 'app.py'
if not real.exists():
    raise FileNotFoundError(f"Dashboard not found at expected location: {real}")

# Ensure repo root is on sys.path so local imports inside the dashboard work
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Execute the real dashboard script
runpy.run_path(str(real), run_name='__main__')
