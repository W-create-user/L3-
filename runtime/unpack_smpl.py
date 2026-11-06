"""Extract only the bundled SMPL source into the expected vendor directory."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'vendor' / 'smpl'
    marker = '/code/vendor/smpl/'
    with zipfile.ZipFile(root / '执行智能体.zip') as archive:
        for entry in archive.infolist():
            if marker not in entry.filename or entry.is_dir():
                continue
            relative = Path(entry.filename.split(marker, 1)[1])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    if not (destination / 'smpl/envs/atropineenv.py').is_file():
        raise RuntimeError('Archive does not contain the expected SMPL source')
    print(destination)


if __name__ == '__main__':
    main()
