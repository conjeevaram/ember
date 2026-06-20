"""Put ``src/`` on sys.path so the entry scripts run without installing the
package (the demo box runs them straight from a checkout)."""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
