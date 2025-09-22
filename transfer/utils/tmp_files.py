import contextlib
import tempfile
from pathlib import Path
from typing import Generator, Optional


@contextlib.contextmanager
def make_named_temporary_file(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    delete: bool = True,
    target_dir: Optional[Path] = None,
) -> Generator[Path, None, None]:
    """Context manager to create a named temporary file.

    Args:
        prefix (Optional[str], optional): Prefix for the file name. Defaults to None.
        delete (bool, optional): If True, the file will be deleted upon exit. Defaults to True.
        target_dir (Optional[Path], optional): Directory where the file should be created. Defaults to None.

    Yields:
        Generator[Path, None, None]: Path of the created temporary file.
    """

    with tempfile.NamedTemporaryFile(
        dir=target_dir, delete=delete, prefix=prefix, suffix=suffix
    ) as file:
        yield Path(file.name)
