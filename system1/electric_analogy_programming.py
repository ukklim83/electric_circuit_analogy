"""Legacy system1 launcher for the shared driver."""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from codes.electric_analogy_programming import main as _shared_main

def main():
    return _shared_main(default_system="system1")

if __name__ == "__main__":
    main()
