"""Загрузка опубликованного кода и весов Magic Leap SuperPoint для учебного запуска."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent / "superpoint"
BASE = "https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/master/"
FILES = {
    "demo_superpoint.py": "613706ae7e9ce3fbc2cfe042fc3f37d739838cc2e1dc8ddd08f8ec037765df04",
    "superpoint_v1.pth": "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
    "LICENSE": "a79449584867091af048b81d4b8fa3a0e2cfa724ccf994a5873dea8c073d9266",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ROOT) as directory:
        for name, expected in FILES.items():
            target = ROOT / name
            if target.is_file() and sha256(target.read_bytes()).hexdigest() == expected:
                continue
            with urlopen(BASE + name, timeout=60) as response:
                data = response.read()
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"Контрольная сумма загруженного файла {name} отличается")
            pending = Path(directory) / name
            pending.write_bytes(data)
            pending.replace(target)
    print("Код, веса и лицензия SuperPoint загружены и проверены.")
    print("Условия использования доступны в superpoint/LICENSE.")


if __name__ == "__main__":
    main()
