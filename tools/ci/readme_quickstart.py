"""Run the README quick start exactly as written.

Used by the clean-install CI job: the package is installed from the built wheel
into a fresh environment, so this catches missing runtime dependencies.
"""

import re
import sys
from pathlib import Path

readme = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md").read_text()
section = readme.split("## Quick start", 1)[1]
code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
print(code)
exec(compile(code, "README quick start", "exec"))
print("README quick start ran.")
