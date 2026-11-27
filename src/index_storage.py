import re
from pathlib import Path
from tempfile import NamedTemporaryFile


def resolve_index_dir(root):
    root = Path(root)
    pointer = root / "CURRENT"
    try:
        name = pointer.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        # Older indexes store files directly under the root.
        return root
    if not re.fullmatch(r"index-[0-9a-f]{32}", name):
        raise ValueError(f"Invalid index generation in {pointer}")
    generation = root / "generations" / name
    if not generation.is_dir():
        raise FileNotFoundError(f"Published index generation is missing: {generation}")
    return generation


def publish_generation(root, generation):
    with NamedTemporaryFile(mode="w", encoding="ascii", dir=root,
                            prefix=".CURRENT-", delete=False) as pointer:
        temporary_path = Path(pointer.name)
        try:
            pointer.write(generation.name + "\n")
            pointer.close()
            temporary_path.replace(root / "CURRENT")
        finally:
            temporary_path.unlink(missing_ok=True)
