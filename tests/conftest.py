import os, sys
# add the repository root to sys.path so `import app` and `import src` work in CI
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
