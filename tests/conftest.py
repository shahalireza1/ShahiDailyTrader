import sys
from pathlib import Path


# Ensure the repository root is on the path for imports during tests.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
