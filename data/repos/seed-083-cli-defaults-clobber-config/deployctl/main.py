import json
from pathlib import Path

from .cli import build_parser
from .settings import resolve


def load_settings(argv, read_file=None):
    read_file = read_file or (lambda path: Path(path).read_text())
    opts = vars(build_parser().parse_args(argv))
    path = opts.pop("config")
    file_cfg = json.loads(read_file(path)) if path else {}
    return resolve(file_cfg, opts)
