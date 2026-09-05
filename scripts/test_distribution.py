"""Install one distribution in a clean environment and test it outside the repo."""

import argparse
import glob
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "distribution", help="Path or glob matching exactly one wheel/sdist"
    )
    args = parser.parse_args()
    artifacts = glob.glob(args.distribution)
    if len(artifacts) != 1:
        parser.error(f"expected one distribution, found {artifacts}")
    artifact = str(Path(artifacts[0]).resolve())
    tests = str(Path(__file__).resolve().parents[1] / "tests" / "python")

    with tempfile.TemporaryDirectory(prefix="sf-limiter-package-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = str(
            environment
            / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        )
        cmd_common = [python, "-I", "-m"]

        cmd = [*cmd_common, "pip", "install", "--no-cache-dir", artifact, "pytest>=8"]
        subprocess.run(cmd, cwd=root, check=True)

        cmd = [*cmd_common, "pytest", "--import-mode=importlib", "-q", tests]
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
