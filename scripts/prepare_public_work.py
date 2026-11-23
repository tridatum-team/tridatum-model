"""Prepare portable, public/seen synthetic diagnostics, never training or holdout admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trid.service.public_work import PROFILE, TASKS, public_row, reference_row, validate_case

SOURCE_SCHEMA = 'trid-public-work-source-v1'
MANIFEST_SCHEMA = 'trid-public-work-suite-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('Non-JSON numeric constant: ' + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _pin(name, data):
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def project_source(source):
    _require(isinstance(source, dict) and set(source) == {'schema', 'scope', 'cases'},
             'Invalid author source fields')
    _require(source['schema'] == SOURCE_SCHEMA and source['scope'] == 'diagnostic',
             'Only synthetic diagnostic sources are supported')
    cases = source['cases']
    _require(isinstance(cases, list) and len(cases) == 24, 'Exactly 24 authored diagnostics required')
    ids, tasks, prompts = set(), set(), set()
    rows, references = [], []
    for case in cases:
        validate_case(case)
        _require(case['id'] not in ids, 'Duplicate case ID: ' + case['id'])
        ids.add(case['id'])
        tasks.add(case['task'])
        row = public_row(case)
        reference = reference_row(case, row)
        _require(reference['prompt_sha256'] not in prompts, 'Duplicate public prompt')
        prompts.add(reference['prompt_sha256'])
        rows.append(row)
        references.append(reference)
    _require(tasks == set(TASKS), 'Missing task coverage: ' + ', '.join(sorted(set(TASKS) - tasks)))
    return rows, references


def verify_manifest(path):
    """Authenticate local pins and exact source projections; not a quality gate."""
    path = Path(path)
    manifest = strict_json(path.read_text(encoding='utf-8'))
    _require(isinstance(manifest, dict) and set(manifest) == {
        'schema', 'profile', 'scope', 'source', 'inputs', 'references', 'expected_tasks',
        'case_count', 'training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen',
    }, 'Invalid manifest fields')
    _require(manifest['schema'] == MANIFEST_SCHEMA and manifest['profile'] == PROFILE
             and manifest['scope'] == 'diagnostic', 'Unsupported suite')
    _require(all(manifest[key] is False for key in (
        'training_ready', 'quality_accepted', 'expert_reviewed', 'historically_unseen',
    )), 'Diagnostic suite cannot claim admission or unseen status')
    _require(manifest['expected_tasks'] == list(TASKS)
             and type(manifest['case_count']) is int and manifest['case_count'] == 24,
             'Invalid expected task/count contract')
    values = {}
    for key, name in (('source', 'source.private.json'), ('inputs', 'inputs.json'),
                      ('references', 'references.private.json')):
        pin = manifest[key]
        _require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                 and pin['path'] == name and type(pin['bytes']) is int,
                 'Invalid local pin: ' + key)
        artifact = path.parent / name
        _require(artifact.is_file() and not artifact.is_symlink(), 'Expected local file: ' + name)
        data = artifact.read_bytes()
        _require(pin == _pin(name, data), 'Artifact pin mismatch: ' + key)
        values[key] = strict_json(data.decode('utf-8'))
    rows, references = project_source(values['source'])
    _require(rows == values['inputs'] and references == values['references'],
             'Pinned artifacts differ from authored projections')
    return manifest


def prepare(source_path, output):
    source_path, output = Path(source_path), Path(output)
    _require(source_path.is_file() and not source_path.is_symlink(), 'Source must be an ordinary file')
    source_bytes = source_path.read_bytes()
    rows, references = project_source(strict_json(source_bytes.decode('utf-8')))
    files = {
        'source.private.json': source_bytes,
        'inputs.json': _bytes(rows),
        'references.private.json': _bytes(references),
    }
    manifest = {
        'schema': MANIFEST_SCHEMA, 'profile': PROFILE, 'scope': 'diagnostic',
        'source': _pin('source.private.json', files['source.private.json']),
        'inputs': _pin('inputs.json', files['inputs.json']),
        'references': _pin('references.private.json', files['references.private.json']),
        'expected_tasks': list(TASKS), 'case_count': len(rows), 'training_ready': False,
        'quality_accepted': False, 'expert_reviewed': False, 'historically_unseen': False,
    }
    # Validate everything before reserving the exclusive destination. No overwrite,
    # timestamp, absolute source path, or implicit tokenizer/model invocation.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, data in files.items():
        with (output / name).open('xb') as handle:
            handle.write(data)
    with (output / 'manifest.json').open('xb') as handle:
        handle.write(_bytes(manifest))
    return verify_manifest(output / 'manifest.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = prepare(args.source, args.output)
    print(json.dumps({'profile': PROFILE, 'scope': manifest['scope'],
                      'case_count': manifest['case_count'],
                      'manifest': str(args.output / 'manifest.json'),
                      'quality_accepted': False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
