import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding=sys.stdout.encoding or "utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding=sys.stderr.encoding or "utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.smarter_brain_checkpoint import main

if __name__ == "__main__":
    raise SystemExit(main())
