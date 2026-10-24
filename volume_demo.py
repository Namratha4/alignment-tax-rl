"""Modal Volume demo: a counter that survives across containers.

Run repeatedly with:  modal run volume_demo.py
"""

import modal
from pathlib import Path

app = modal.App("alignment-tax")

vol = modal.Volume.from_name("al-tax", create_if_missing=True)

@app.function(volumes={"/vol":vol})
def bump():
  path = Path("/vol/counter.txt")

  if path.exists():
    value = int(path.read_text())
  else:
    value = 0
  value+=1

  path.write_text(str(value))
  # vol.commit()

  return value

@app.local_entrypoint()
def main():
  print(bump.remote())
