"""Execute every cell in a clean native Python kernel before delivery."""
import asyncio
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
root = Path(__file__).resolve().parents[1]
path = root/"content"/"kompot.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=240, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb, path)
print(f"Executed and validated {len(nb.cells)} cells")
