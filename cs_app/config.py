"""Application paths and private process-local credentials; no startup acquisition."""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_key(env_file: Path | None = None) -> str:
    key = os.environ.get('ODDSPAPI_API_KEY', '').strip()
    if key or env_file is None:
        return key
    values = []
    if not env_file.is_file():
        raise ValueError('Explicit credential environment file does not exist')
    for line in env_file.read_text().splitlines():
        cleaned = line.strip()
        if cleaned.startswith('export '):
            cleaned = cleaned[7:].strip()
        if not cleaned.startswith('ODDSPAPI_API_KEY='):
            continue
        tokens = shlex.split(cleaned.split('=', 1)[1], comments=True)
        if len(tokens) != 1:
            raise ValueError('Credential environment assignment must have one quoted or unquoted value')
        values.append(tokens[0])
    if len(values) > 1:
        raise ValueError('Duplicate credential environment assignment')
    return values[0] if values else ''


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=lambda: REPO / 'data/app')
    repo: Path = field(default_factory=lambda: REPO)
    key: str = field(default='', repr=False)

    @classmethod
    def from_args(cls, data_dir: Path | None = None, env_file: Path | None = None):
        return cls((data_dir or REPO / 'data/app').resolve(), REPO, load_key(env_file))
