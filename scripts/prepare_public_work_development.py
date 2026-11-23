"""Package fictional development candidates; admit only after two packet-bound reviews.

Neither preparation nor annotation admission certifies expert or model quality.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from check_reserved_overlap import exact_hash, ngrams
from prepare_public_work import (SOURCE_SCHEMA, MANIFEST_SCHEMA, _bytes, _pin,
                                 _require as require, project_source, strict_json)
from trid.service.public_work import GENERAL_TASKS, PROFILE, TASKS, public_row, reference_row, prompt_sha256

BATCH_SCHEMA = 'trid-public-work-development-batch-v1'
PREPARED_SCHEMA = 'trid-public-work-development-prepared-v1'
TRAINING_SCHEMA = 'trid-public-work-training-manifest-v1'
PACKET_SCHEMA = 'trid-public-work-annotation-packet-v1'
REVIEW_SCHEMA = 'trid-public-work-annotation-review-v1'
GENERAL_INSTITUTION = '해당 없음(가상)'
FALSE_FLAGS = {'expert_certified': False, 'semantic_evaluated': False}
NUMBER = re.compile(r'-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected ordinary file: ' + str(path))
    return path.read_bytes()


def read_json(path):
    return strict_json(read_file(path).decode('utf-8'))


def arithmetic(expression):
    """Interpret only decimal constants, arithmetic, and min/max; never eval code."""
    require(isinstance(expression, str) and expression.strip()
            and re.fullmatch(r'[0-9\s.+*/(),a-z-]+', expression), 'Invalid arithmetic expression')
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except (SyntaxError, ValueError) as error:
        raise ValueError('Invalid arithmetic expression') from error
    expression = expression.strip()

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            require(NUMBER.fullmatch(literal) is not None, 'Only decimal arithmetic literals allowed')
            return Fraction(literal)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            require(right != 0, 'Arithmetic division by zero')
            return left / right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ('min', 'max') and node.args and not node.keywords):
            values = [visit(arg) for arg in node.args]
            return min(values) if node.func.id == 'min' else max(values)
        raise ValueError('Unsupported arithmetic syntax')

    return visit(tree.body)


def validate_calculation(case):
    calculation = case['rubric']['calculation']
    require(calculation is not None or case['task'] not in ('table_calculation', 'general_arithmetic'),
            'Calculation required: ' + case['id'])
    if calculation is None:
        return
    result = calculation['result']
    require(NUMBER.fullmatch(result) is not None, 'Invalid calculation result: ' + case['id'])
    expected = Fraction(result)
    require(arithmetic(calculation['expression']) == expected, 'Calculation mismatch: ' + case['id'])
    tokens = re.findall(r'(?<![0-9.,])-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?![0-9.,])',
                        case['target'])
    require(any(Fraction(token.replace(',', '')) == expected for token in tokens),
            'Calculation result missing from target: ' + case['id'])


def load_batch(path):
    batch = read_json(path)
    require(isinstance(batch, dict) and set(batch) == {
        'schema', 'batch', 'split', 'business_group', 'institutions', 'cases'}, 'Invalid batch fields')
    require(batch['schema'] == BATCH_SCHEMA and batch['split'] in ('train', 'holdout'), 'Invalid batch schema/split')
    name = batch['batch']
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_-]+', name), 'Invalid batch name')
    require(isinstance(batch['business_group'], str) and batch['business_group'].strip(), 'Missing business group')
    institutions = batch['institutions']
    require(isinstance(institutions, list)
            and all(isinstance(item, str) and item.startswith('가상 ') for item in institutions)
            and len(set(institutions)) == len(institutions), 'Invalid declared institutions')
    require(isinstance(batch['cases'], list) and batch['cases'], 'Empty batch')
    ids = set()
    for case in batch['cases']:
        row = public_row(case)
        reference_row(case, row)
        require(case['id'].startswith('dev-' + name + '-') and case['id'] not in ids, 'Invalid/duplicate batch case ID')
        ids.add(case['id'])
        require(all(family.startswith(name + '-') for family in case['family_ids']), 'Invalid batch family prefix')
        require(case['provenance']['kind'] == 'synthetic_development', 'Development provenance required')
        require(case['institution'] == GENERAL_INSTITUTION if case['task'] in GENERAL_TASKS
                else case['institution'] in institutions, 'Undeclared case institution')
        require(all(document['institution'] in institutions for document in case['documents']),
                'Undeclared document institution')
        validate_calculation(case)
    return batch


def identities(cases, declared=()):
    result = {key: set() for key in ('institutions', 'family_ids', 'case_ids', 'prompt_sha256', 'documents', 'questions')}
    result['institutions'].update(declared)
    documents = []
    for case in cases:
        result['institutions'].add(case['institution'])
        result['family_ids'].update(case['family_ids'])
        result['case_ids'].add(case['id'])
        result['prompt_sha256'].add(prompt_sha256(public_row(case)['messages']))
        result['questions'].add(exact_hash(case['question']))
        for document in case['documents']:
            result['institutions'].add(document['institution'])
            result['documents'].add(exact_hash(document['text']))
            documents.append((case['id'], ngrams(document['text'])))
    result['institutions'].discard(GENERAL_INSTITUTION)
    return result, documents


def check_separation(splits, diagnostics, declared):
    # Diagnostics are used only to derive identity/overlap fingerprints. No diagnostic
    # question, target, rubric, or document is emitted into any prepared artifact.
    indexes = {name: identities(cases, declared.get(name, ()))
               for name, cases in {**splits, 'diagnostics': diagnostics}.items()}
    for left, right in combinations(indexes, 2):
        a, adocs = indexes[left]
        b, bdocs = indexes[right]
        for key in a:
            require(not a[key] & b[key], f'Split overlap ({key}): {left}/{right}')
        for aid, grams in adocs:
            if len(grams) < 20:
                continue
            for bid, other in bdocs:
                shared = len(grams & other)
                require(not (shared >= 20 and shared / min(len(grams), len(other)) >= 0.5),
                        f'Near-duplicate documents: {left}/{right}: {aid}/{bid}')


def suite_files(cases):
    source = {'schema': SOURCE_SCHEMA, 'scope': 'development_holdout', 'cases': cases}
    rows, references = project_source(source)
    files = {'source.private.json': _bytes(source), 'inputs.json': _bytes(rows),
             'references.private.json': _bytes(references)}
    manifest = {'schema': MANIFEST_SCHEMA, 'profile': PROFILE, 'scope': source['scope'],
                **{key: _pin(name, files[name]) for key, name in
                   (('source', 'source.private.json'), ('inputs', 'inputs.json'), ('references', 'references.private.json'))},
                'expected_tasks': list(TASKS), 'case_count': len(cases), 'training_ready': False,
                'quality_accepted': False, 'expert_reviewed': False, 'historically_unseen': False}
    files['manifest.json'] = _bytes(manifest)
    return {'holdout/' + name: data for name, data in files.items()}


def expert_packet(cases):
    lines = ['# 공공업무 개발 홀드아웃 전문가 검토 자료', '',
             '**아직 검토되지 않았습니다. AI 작성 가상 자료이며 전문가 정답·미노출 시험이 아닙니다.**', '',
             '제안 답변을 정답으로 간주하지 말고 문서 근거와 계산을 독립적으로 확인해 주세요.', '']
    verdicts = []
    for case in cases:
        lines.extend(['## ' + case['id'], '', '과제: ' + case['task'], '기관: ' + case['institution'], '', '### 제공 문서'])
        for document in case['documents']:
            lines.extend(['', f"#### [{document['id']}] {document['title']} ({document['institution']})", '', document['text']])
        if not case['documents']:
            lines.append('제공 문서 없음 (일반 과제)')
        if case['history']:
            lines.extend(['', '### 이전 대화', '', json.dumps(case['history'], ensure_ascii=False, indent=2)])
        lines.extend(['', '### 질문', '', case['question'], '', '### 출력 계약', '',
                      json.dumps(case['output_contract'], ensure_ascii=False, indent=2), '', '### 제안 답변 (미검토)', '',
                      case['target'], '', '### 평가 기준 (미검토)', '', json.dumps(case['rubric'], ensure_ascii=False, indent=2),
                      '', '판정 (verdict):', '문제 (issue):', '교정 답변 (corrected_answer):', '검토자 (reviewer):', '날짜 (date):', ''])
        verdicts.append({'id': case['id'], 'verdict': '', 'issue': '', 'corrected_answer': '', 'reviewer': '', 'date': ''})
    template = {'schema': 'trid-public-work-expert-review-template-v1', 'scope': 'development_holdout',
                'status': 'NOT_REVIEWED', 'expert_reviewed': False, 'cases': verdicts}
    return '\n'.join(lines).encode('utf-8'), template


def training_manifest(identifier, train_pin, rows, *, admitted=False):
    return {'schema': TRAINING_SCHEMA, 'profile': PROFILE, 'id': identifier,
            'state': 'ADMITTED_FOR_RESEARCH' if admitted else 'CANDIDATE',
            'training_ready': admitted, **FALSE_FLAGS, 'rows': rows,
            'train': [{**train_pin, 'adapter': 'grounded_messages', 'epochs': 1, 'split': 'train'}], 'valid': []}


def write_package(output, files, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, data in {**files, 'manifest.json': _bytes(manifest)}.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open('xb') as handle:
            handle.write(data)


def prepare(batches, diagnostics, output):
    paths = sorted(Path(batches).glob('*.json'))
    require(paths, 'No development batches found')
    batches_loaded = [load_batch(path) for path in paths]
    require(len({batch['batch'] for batch in batches_loaded}) == len(batches_loaded), 'Duplicate batch name')
    splits = {split: [case for batch in batches_loaded if batch['split'] == split for case in batch['cases']]
              for split in ('train', 'holdout')}
    require(all(splits.values()), 'Both train and holdout batches required')
    for split, cases in splits.items():
        tasks = {case['task'] for case in cases}
        require(tasks == set(TASKS), 'Missing task coverage in split ' + split + ': '
                + ', '.join(sorted(set(TASKS) - tasks)))
    all_cases = splits['train'] + splits['holdout']
    require(len({case['id'] for case in all_cases}) == len(all_cases), 'Duplicate development case ID')
    diagnostic_source = read_json(diagnostics)
    require(isinstance(diagnostic_source, dict) and diagnostic_source.get('schema') == SOURCE_SCHEMA
            and diagnostic_source.get('scope') == 'diagnostic'
            and isinstance(diagnostic_source.get('cases'), list) and diagnostic_source['cases'], 'Invalid diagnostic source')
    declared = {split: {institution for batch in batches_loaded if batch['split'] == split
                        for institution in batch['institutions']} for split in splits}
    check_separation(splits, diagnostic_source['cases'], declared)
    rows = [{'id': case['id'], 'task': 'public_work_' + case['task'], 'split': 'train',
             'messages': public_row(case)['messages'], 'answer': case['target']} for case in splits['train']]
    train = b''.join((json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode() for row in rows)
    packet = {'schema': PACKET_SCHEMA, 'profile': PROFILE, 'state': 'NOT_REVIEWED', **FALSE_FLAGS,
              'batches': batches_loaded}
    packet_sha = canonical_sha(packet)
    files = {'train.jsonl': train, 'review-packet.json': _bytes(packet), **suite_files(splits['holdout'])}
    candidate = training_manifest('public-work-development-' + packet_sha[:16], _pin('train.jsonl', train), len(rows))
    files['training-manifest.json'] = _bytes(candidate)
    markdown, template = expert_packet(splits['holdout'])
    template['packet_sha256'] = packet_sha
    files['expert-review/holdout.md'] = markdown
    files['expert-review/verdicts.template.json'] = _bytes(template)
    counts = {}
    for split, cases in splits.items():
        tasks = Counter(case['task'] for case in cases)
        lengths = [len(document['text']) for case in cases for document in case['documents']]
        counts[split] = {'rows': len(cases), 'task': dict(sorted(tasks.items())),
                         'institution': dict(sorted(Counter(case['institution'] for case in cases).items())),
                         'business_group': dict(sorted(Counter(batch['business_group'] for batch in batches_loaded
                                                               if batch['split'] == split for _ in batch['cases']).items())),
                         'general_task_share': sum(tasks[task] for task in GENERAL_TASKS) / len(cases),
                         'document_length_chars': {'count': len(lengths), 'min': min(lengths), 'max': max(lengths),
                                                   'mean': sum(lengths) / len(lengths), 'median': median(lengths)}}
    manifest = {'schema': PREPARED_SCHEMA, 'profile': PROFILE, 'state': 'CANDIDATE', 'training_ready': False,
                **FALSE_FLAGS, 'quality_accepted': False, 'expert_reviewed': False, 'historically_unseen': False,
                'packet_sha256': packet_sha, 'counts': counts,
                'batch_task_counts': {batch['batch']: dict(sorted(Counter(case['task'] for case in batch['cases']).items()))
                                      for batch in batches_loaded},
                'artifacts': {name: _pin(name, data) for name, data in files.items()},
                'builder': _pin(str(Path(__file__).resolve()), read_file(__file__)),
                'batch_sources': [_pin(str(path.resolve()), read_file(path)) for path in paths],
                'diagnostics': _pin(str(Path(diagnostics).resolve()), read_file(diagnostics)),
                'separation': {'exact_normalization': 'NFC, whitespace removed', 'word_ngram': 8,
                               'minimum_shared_grams': 20, 'containment': 0.5, 'passed': True}}
    write_package(output, files, manifest)
    return manifest


def authenticate_prepared(path):
    path = Path(path)
    manifest = read_json(path)
    require(manifest.get('schema') == PREPARED_SCHEMA and manifest.get('profile') == PROFILE
            and manifest.get('state') == 'CANDIDATE' and manifest.get('training_ready') is False
            and all(manifest.get(key) is False for key in FALSE_FLAGS), 'Invalid candidate manifest')
    artifacts = manifest.get('artifacts')
    require(isinstance(artifacts, dict) and {'train.jsonl', 'review-packet.json', 'training-manifest.json'} <= artifacts.keys(),
            'Missing candidate artifacts')
    loaded = {}
    for name, pin in artifacts.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name,
                'Unsafe artifact path')
        local = path.parent / relative
        require(local.resolve().is_relative_to(path.parent.resolve()), 'Artifact escapes candidate directory')
        data = read_file(local)
        require(pin == _pin(name, data), 'Candidate artifact pin mismatch: ' + name)
        loaded[name] = data
    packet = strict_json(loaded['review-packet.json'].decode('utf-8'))
    require(packet.get('schema') == PACKET_SCHEMA and canonical_sha(packet) == manifest.get('packet_sha256'),
            'Candidate packet hash mismatch')
    candidate = strict_json(loaded['training-manifest.json'].decode('utf-8'))
    require(candidate == training_manifest(candidate.get('id'), _pin('train.jsonl', loaded['train.jsonl']),
                                           manifest['counts']['train']['rows']), 'Candidate training manifest mismatch')
    cases = [case for batch in packet['batches'] for case in batch['cases']]
    require(cases and len({case['id'] for case in cases}) == len(cases), 'Invalid packet case inventory')
    expected_rows = [{'id': case['id'], 'task': 'public_work_' + case['task'], 'split': 'train',
                      'messages': public_row(case)['messages'], 'answer': case['target']}
                     for batch in packet['batches'] if batch['split'] == 'train' for case in batch['cases']]
    actual_rows = [strict_json(line) for line in loaded['train.jsonl'].decode('utf-8').splitlines()]
    require(actual_rows == expected_rows and len(actual_rows) == candidate['rows'], 'Candidate train projection mismatch')
    return manifest, candidate, cases, loaded['train.jsonl']


def admit(prepared, reviews, output):
    require(len(reviews) == 2, 'Exactly two independent annotation reviews required')
    manifest, candidate, cases, train = authenticate_prepared(prepared)
    expected = {case['id'] for case in cases}
    reviewers, pins = set(), []
    for path in reviews:
        data = read_file(path)
        review = strict_json(data.decode('utf-8'))
        require(isinstance(review, dict) and set(review) == {
            'schema', 'packet_sha256', 'reviewer', 'model_outputs_seen', 'expert_certified', 'cases'}, 'Invalid review fields')
        require(review['schema'] == REVIEW_SCHEMA and review['packet_sha256'] == manifest['packet_sha256']
                and review['model_outputs_seen'] is False and review['expert_certified'] is False, 'Invalid review binding/status')
        reviewer = review['reviewer']
        require(isinstance(reviewer, str) and reviewer.strip() and reviewer.strip() not in reviewers,
                'Distinct nonempty reviewers required')
        reviewers.add(reviewer.strip())
        require(isinstance(review['cases'], list), 'Review cases must be an array')
        seen = set()
        for verdict in review['cases']:
            require(isinstance(verdict, dict) and set(verdict) == {'id', 'pass', 'reason'}, 'Invalid annotation verdict')
            require(isinstance(verdict['id'], str) and verdict['id'] in expected and verdict['id'] not in seen,
                    'Unknown/duplicate reviewed case')
            seen.add(verdict['id'])
            require(verdict['pass'] is True and isinstance(verdict['reason'], str) and verdict['reason'].strip(),
                    'Every case must pass with a nonempty reason')
        require(seen == expected, 'Incomplete annotation review')
        pins.append(_pin(str(Path(path).resolve()), data))
    admitted = training_manifest(candidate['id'], _pin('train.jsonl', train), candidate['rows'], admitted=True)
    admitted.update(candidate_manifest=_pin(str(Path(prepared).resolve()), read_file(prepared)),
                    reviews=pins, packet_sha256=manifest['packet_sha256'])
    write_package(output, {'train.jsonl': train}, admitted)
    return admitted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--batches', required=True, type=Path)
    preparation.add_argument('--diagnostics', required=True, type=Path)
    preparation.add_argument('--output', required=True, type=Path)
    admission = commands.add_parser('admit')
    admission.add_argument('--prepared', required=True, type=Path)
    admission.add_argument('--review', required=True, action='append', type=Path)
    admission.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = (prepare(args.batches, args.diagnostics, args.output) if args.command == 'prepare'
              else admit(args.prepared, args.review, args.output))
    print(json.dumps({'manifest': str(args.output / 'manifest.json'), 'state': result['state'],
                      'training_ready': result['training_ready'], 'expert_certified': False,
                      **({'counts': result['counts'], 'batch_task_counts': result['batch_task_counts']}
                         if args.command == 'prepare' else {'rows': result['rows']})}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
