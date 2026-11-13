"""Installation settings for the on-prem server (JSON file, unknown keys rejected)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from trid.pdf_isolated import PdfLimits


@dataclass(frozen=True)
class Config:
    hostname: str
    port: int = 8080
    data_dir: Path = Path('/data')
    llm_base: str = 'http://llm:8080'
    tokenizer_path: Path = Path('/opt/trid/tokenizer.json')
    retention_days: int = 365
    pdf: PdfLimits = PdfLimits()
    max_question: int = 2000
    idle_seconds: int = 1800
    max_sessions: int = 32
    slots: int = 8
    per_minute: int = 20
    lock_wait: float = 2.0
    context: int = 4096
    max_tokens: int = 512
    family: str = 'qwen3'


_PATHS = {'data_dir', 'tokenizer_path'}


def _typed(name: str, value, default):
    if isinstance(default, bool) or isinstance(value, bool):
        raise ValueError(f'설정 값 형식이 올바르지 않습니다: {name}')
    if isinstance(default, float):
        ok = isinstance(value, (int, float))
    elif isinstance(default, int):
        ok = isinstance(value, int)
    else:
        ok = isinstance(value, str)
    if not ok:
        raise ValueError(f'설정 값 형식이 올바르지 않습니다: {name}')
    return value


def load_config(path: Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('설정 파일은 JSON 객체여야 합니다.')
    known = {f.name: f for f in fields(Config)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f'알 수 없는 설정 키: {sorted(unknown)}')
    if not isinstance(raw.get('hostname'), str) or not raw['hostname']:
        raise ValueError('hostname 설정이 필요합니다.')
    values = {}
    for name, value in raw.items():
        if name == 'pdf':
            if not isinstance(value, dict):
                raise ValueError('pdf 설정은 객체여야 합니다.')
            allowed = {f.name: f for f in fields(PdfLimits)}
            bad = set(value) - set(allowed)
            if bad:
                raise ValueError(f'알 수 없는 pdf 설정 키: {sorted(bad)}')
            base = PdfLimits()
            values[name] = PdfLimits(**{k: _typed(k, v, getattr(base, k)) for k, v in value.items()})
        elif name in _PATHS:
            values[name] = Path(_typed(name, value, ''))
        else:
            values[name] = _typed(name, value, getattr(Config, name, '') if name != 'hostname' else '')
    if 'retention_days' in values and values['retention_days'] < 1:
        raise ValueError('retention_days는 1 이상이어야 합니다.')
    return Config(**values)
