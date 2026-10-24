"""Step 2: Modal Volume demo -- a counter that survives across containers."""

# TODO 1: import modal, os; create the App
# TODO 2: create the Volume with modal.Volume.from_name(..., create_if_missing=True)
# TODO 3: bump() -- mounted at /vol; read /vol/counter.txt (missing -> 0), add 1,
#         write back, vol.commit(), return the new value
# TODO 4: local entrypoint that calls bump.remote() and prints the result
#
# Run 3 times with:  modal run volume_demo.py

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
  