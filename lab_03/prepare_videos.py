"""Подготовка двух AVI без звука из загруженных MOV с частотой 30 кадров/с."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


LAB_DIR = Path(__file__).resolve().parent


def main():
    videos = LAB_DIR / "videos"
    sources = [videos / f"{number:02d}.MOV" for number in (1, 2)]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    with TemporaryDirectory(dir=videos) as directory:
        for source in sources:
            destination = Path(directory) / source.with_suffix(".avi").name
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(source), "-map", "0:v:0", "-an", "-vf", "fps=30",
                "-c:v", "mpeg4", "-q:v", "3", str(destination),
            ], check=True)
        for source in sources:
            name = source.with_suffix(".avi").name
            (Path(directory) / name).replace(videos / name)
    print("Подготовлены videos/01.avi и videos/02.avi без звука, 30 кадров/с.")


if __name__ == "__main__":
    main()
