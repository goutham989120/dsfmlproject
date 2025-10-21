from pathlib import Path
import runpy
import os
import sys

# This wrapper allows running Streamlit from notebook/data using:
#   streamlit run dashboard/app.py
# It locates the project root (three levels up from this file), changes
# the current working directory to the project root, and executes the
# real dashboard script at dashboard/app.py.

THIS_FILE = Path(__file__).resolve()
# notebook/data/dashboard -> parents[0]=notebook/data/dashboard, [1]=notebook/data, [2]=notebook, [3]=<project root>
PROJECT_ROOT = THIS_FILE.parents[3]
TARGET = PROJECT_ROOT / 'dashboard' / 'app.py'

if not TARGET.exists():
    msg = f"Dashboard entrypoint not found at: {TARGET}"
    raise FileNotFoundError(msg)

# Set CWD to project root so the dashboard script resolves relative paths
os.chdir(str(PROJECT_ROOT))

# Ensure project root is on sys.path so imports resolve the same as when running from project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute the real dashboard script
runpy.run_path(str(TARGET), run_name='__main__')
