"""Build an ordinary static JupyterLite distribution for any static host."""
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "jupyterlite_core.app", "build", "--contents", "content",
                "--output-dir", "_output"], cwd=root, check=True)
# JupyterLite already provides index.html and lab/index.html. Keep those entrypoints.
(root/"_output"/".nojekyll").touch()
report = root/"output"/"kompot-report.html"
if report.exists():
    shutil.copy2(report, root/"_output"/"report.html")
print("Built: _output/lab/index.html?path=kompot.ipynb")
