"""Obtain the course inputs without overwriting a different existing file."""
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip"
EXPECTED = {
    "reflector.json": "22b84948e28e8e13e958f47b2d4bf9a70a3d0c8ee6ffcaab3cdeb8003b9902a6",
    "marmousi.json": "67beabdee96d8e3f6433c2f897b3b24b0c744e7808784148bb06310665d888b0",
}


def main():
    missing = []
    for name, expected in EXPECTED.items():
        path = ROOT / "inputs" / name
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"Existing input differs: {path}; retained unchanged.")
            print(f"verified {name}: {actual}")
        else:
            missing.append(name)
    if not missing:
        return
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in missing:
            matches = [member for member in archive.namelist() if Path(member).name == name]
            if len(matches) != 1:
                raise SystemExit(f"Expected exactly one {name} in course archive")
            content = archive.read(matches[0])
            actual = hashlib.sha256(content).hexdigest()
            if actual != EXPECTED[name]:
                raise SystemExit(f"Downloaded input fingerprint differs: {name}")
            path = ROOT / "inputs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(content)
            print(f"downloaded and verified {name}: {actual}")


if __name__ == "__main__":
    main()
