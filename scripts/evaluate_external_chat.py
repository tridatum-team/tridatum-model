"""Compare a local llama.cpp chat model on unchanged supplied-evidence QA sets (RAG310, institution probe).

Comparison-only: no model enters the trid product. Uses the native chat template,
greedy decoding, 64 output tokens and disabled thinking. Preregistered prompt conditions
are reported separately; none is selected using evaluation scores. `evidence_constrained`
uses the grammar_instructed prompt plus a per-item llama.cpp GBNF grammar that only admits
`확인 불가`, `근거 충돌`, or one/two `span [Dn]` answers copied from passage Dn
(trid.span_scoring candidates): the product copy guarantee, enforced at decode time.
Legacy scoring is preserved; strict_product additionally requires full output grammar.
content_* rescoring accepts the source-first order `[Dn] value` as well, so reading ability is not hidden
by citation order; mentions_gold only asks whether every gold value (or the abstention phrase) appears.
This measures supplied-evidence QA, not retrieval, security or general model parity.
`--freeform-prompts` instead records 256-token answers to stored prompts for human/assistant review, unscored.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_rag_eval import normalized, score
from trid.llama_client import QWEN3_NO_THINK, request, complete  # re-exported for other scripts
from trid.prompts import rag_passages
from trid.span_scoring import candidates

SYSTEM = ('주어진 근거만 사용하세요. 답은 값 [Dn] 형식으로 쓰고, 여러 값은 쉼표로 구분하세요. '
          '근거가 없으면 확인 불가, 근거가 충돌하면 근거 충돌만 출력하세요. 설명은 덧붙이지 마세요.')
GRAMMAR = re.compile(r'([^\[\]]+?)\s*\[(D[1-9])\](?:\s*,\s*([^\[\]]+?)\s*\[(D[1-9])\])?')
SOURCE_FIRST = re.compile(r'\[(D[1-9])\]\s*([^\[\],\n]+)')
MAX_TOKENS = 64
FREEFORM_MAX_TOKENS = 256
QWEN3_NO_THINK = '<think>\n\n</think>\n\n'
SCORED = ('strict', 'strict_product', 'value', 'lenient_value', 'source', 'abstained',
          'content_strict', 'content_value', 'mentions_gold', 'exact')
CONDITIONS = {'prompt_only': (None, False), 'grammar_instructed': (SYSTEM, False),
              'evidence_constrained': (SYSTEM, True)}


def gbnf_literal(text):
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"'


def evidence_grammar(prompt):
    """GBNF accepting exactly the product grammar over this prompt's evidence: an abstention, or one or two
    `span [Dn]` answers whose spans are trid.span_scoring candidates (word-boundary substrings of passage Dn).
    Candidates are compiled into a compressed character trie so llama.cpp tracks only matching prefixes."""
    answers = sorted({text for text, span, _ in candidates(rag_passages(prompt)) if span is not None})
    trie = {}
    for text in answers:
        node = trie
        for ch in text:
            node = node.setdefault(ch, {})
        node[''] = {}
    rules = []

    def compile_node(node):
        alternatives = []
        for ch, child in sorted(node.items()):
            if ch == '':
                continue
            label = ch
            while len(child) == 1 and '' not in child:
                (next_ch, child), = child.items()
                label += next_ch
            if child == {'': {}}:
                alternatives.append(gbnf_literal(label))
            else:
                alternatives.append(f'{gbnf_literal(label)} {compile_node(child)}')
        name = f'n{len(rules)}'
        rules.append(f'{name} ::= ({" | ".join(alternatives)})' + ('?' if '' in node else ''))
        return name

    root = compile_node(trie) if trie else None
    value = f'{root} (", " {root})?' if root else '"확인 불가"'
    return '\n'.join([f'root ::= "확인 불가" | "근거 충돌" | {value}'] + rules)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def format_ok(text):
    text = text.strip()
    if text in ('확인 불가', '근거 충돌'):
        return True
    m = GRAMMAR.fullmatch(text)
    return bool(m and m[1].strip() and (m[3] is None or m[3].strip()))


def product_score(row, generated):
    result = score(row, generated)
    result.update(format_ok=format_ok(generated))
    result['strict_product'] = result['strict'] and result['format_ok']
    result.update(content_score(row, generated))
    # Institution probe metric (trid reader's `exact`): raw answer is one of the declared accepted strings.
    result['exact'] = generated.strip() in row.get('accepted_answers', ())
    return result


def content_score(row, generated):
    """Order-free value/source check on the first line, plus a loose gold-mention check."""
    first = generated.strip().split('\n', 1)[0]
    swapped = ', '.join(f'{value.strip()} [{source}]' for source, value in SOURCE_FIRST.findall(first))
    scores = [score(row, first)] + ([score(row, swapped)] if swapped else [])
    pred = normalized(generated)
    if row['gold']:
        mentions = all(any(normalized(v) in pred for v in variants) for variants in row['gold'])
    else:
        mentions = normalized(row['answer']) in pred
    return dict(content_strict=any(s['strict'] for s in scores), content_value=any(s['value'] for s in scores),
                mentions_gold=mentions)


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-radius, center+radius]


def summarize(rows, results):
    counts = defaultdict(Counter)
    known = defaultdict(Counter)
    for row, result in zip(rows, results, strict=True):
        for key in ('all', row['origin'], row['task'], 'answerable' if row['gold'] else 'unanswerable'):
            c = counts[key]
            c['total'] += 1
            for field in ('strict', 'strict_product', 'value', 'lenient_value', 'source', 'abstained',
                          'content_strict', 'content_value', 'mentions_gold', 'exact',
                          'skipped_context', 'truncated_output', 'context_truncated'):
                c[field] += bool(result.get(field, False))
                known[key][field] += field in result
            c['errors'] += result.get('exception') is not None
            known[key]['errors'] += 'exception' in result
            c['format_error'] += result.get('generated') is not None and not result['format_ok']
    summary = {}
    for key, value in sorted(counts.items()):
        entry = dict(value)
        for field in ('skipped_context', 'truncated_output', 'context_truncated', 'errors'):
            if known[key][field] != value['total']:
                entry[field] = None
        entry['strict_product_wilson95'] = wilson(value['strict_product'], value['total'])
        summary[key] = entry
    return summary


def freeform(base, family, context, path, output, model_hash):
    """Stored prompts (e.g. reports/public-dialogue-01-after.json results) -> unscored raw answers.

    Each answer is journaled to `<output>.partial.jsonl` as it finishes (scripts/eval_journal.py); rerunning the
    same command resumes after an interruption, and the report is written only when every prompt id is present."""
    from eval_journal import Journal
    prompts = json.loads(path.read_text())['results']
    journal = Journal(output, dict(model_sha256=model_hash, family=family, prompts_sha256=digest(path),
                                   max_output_tokens=FREEFORM_MAX_TOKENS, context=context))
    for item in prompts:
        if journal.done(item['id']):
            continue
        result = dict(id=item['id'], kind=item.get('kind'), prompt=item['prompt'], reference=item.get('reference'),
                      generated=None, exception=None, skipped_context=False)
        started = time.perf_counter()
        try:
            result.update(complete(base, [{'role': 'user', 'content': item['prompt']}], family, FREEFORM_MAX_TOKENS, context))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['exception'] = f'{type(exc).__name__}: {exc}'
        result['seconds'] = time.perf_counter()-started
        journal.append(result)
    results, counts = journal.finalize([p['id'] for p in prompts])
    return dict(prompts_path=str(path), prompts_sha256=digest(path), max_output_tokens=FREEFORM_MAX_TOKENS,
                scoring='none; review raw answers', counts=counts, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18473')
    parser.add_argument('--model-file', required=True, type=Path)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--family', choices=['qwen3', 'generic'], default='qwen3',
                        help='qwen3 additionally requires the rendered no-thinking prefix')
    parser.add_argument('--data', type=Path, default=Path('data/korean-rag-eval-v1'))
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--freeform-prompts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1') or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        parser.error('Use a literal loopback HTTP llama.cpp server; remote document upload is not supported')
    if args.output.exists():
        parser.error('Use a new output file')
    base = args.url.rstrip('/')
    props = request(base, '/props')
    if Path(props['model_path']).resolve() != args.model_file.resolve():
        raise ValueError('Server model does not match --model-file')
    model_hash = digest(args.model_file)
    context = props['default_generation_settings']['n_ctx']
    report = dict(protocol=__doc__, model_id=args.model_id, revision=args.revision, family=args.family,
                  model_sha256=model_hash, model_bytes=args.model_file.stat().st_size, runtime=props,
                  temperature=0, thinking=False, evaluated_at_unix=time.time())
    if args.freeform_prompts:
        report.update(freeform(base, args.family, context, args.freeform_prompts, args.output, model_hash))
        if digest(args.model_file) != model_hash:
            raise ValueError('Model changed during evaluation')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f'{len(report["results"])} freeform answers -> {args.output}')
        return
    test = args.data/'test.jsonl'
    test_hash = digest(test)
    provenance = json.loads((args.data/'provenance.json').read_text())
    if test_hash != provenance['hashes']['test']:
        raise ValueError('Evaluation data changed')
    rows = [json.loads(line) for line in test.read_text().splitlines()]
    report.update(data=str(args.data), test_sha256=test_hash, max_output_tokens=MAX_TOKENS, conditions={}, comparisons=[])
    # Conditions are fixed before any evaluation item is generated.
    for condition in args.conditions:
        system, constrained = CONDITIONS[condition]
        results = []
        for i, row in enumerate(rows):
            result = dict(id=row['id'], task=row['task'], origin=row['origin'], answer=row['answer'],
                          generated=None, prompt_tokens=None, output_tokens=None, finish_reason=None,
                          exception=None, skipped_context=False, context_truncated=False,
                          truncated_output=False, format_ok=False)
            started = time.perf_counter()
            try:
                messages = ([] if system is None else [{'role': 'system', 'content': system}])
                messages.append({'role': 'user', 'content': row['prompt']})
                result.update(complete(base, messages, args.family, MAX_TOKENS, context,
                                       evidence_grammar(row['prompt']) if constrained else None))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result['exception'] = f'{type(exc).__name__}: {exc}'
            if i == 0 and result['exception']:
                raise RuntimeError(f'{condition}: first inference failed: {result["exception"]}')
            result['seconds'] = time.perf_counter()-started
            result.update(product_score(row, result['generated'] or ''))
            if result['exception'] or result['skipped_context'] or result['context_truncated']:
                for metric in SCORED:
                    result[metric] = False
            results.append(result)
            if (i+1) % 25 == 0:
                print(f'{condition}: {i+1}/{len(rows)}', flush=True)
        report['conditions'][condition] = dict(system_prompt=system,
            system_prompt_sha256=hashlib.sha256((system or '').encode()).hexdigest(),
            summary=summarize(rows, results), results=results)
    if digest(args.model_file) != model_hash or digest(test) != test_hash:
        raise ValueError('Model or evaluation data changed during evaluation')
    # Rescore the exact stored raw answers (trid or earlier external runs), with no model-specific text cleanup.
    for path in args.compare_report:
        previous = json.loads(path.read_text())
        previous_results = previous['results'] if 'results' in previous else previous['conditions']['grammar_instructed']['results']
        if previous['test_sha256'] != test_hash or [r['id'] for r in previous_results] != [r['id'] for r in rows]:
            raise ValueError(f'Comparison must contain the same complete ordered dataset: {path}')
        results = []
        for row, old in zip(rows, previous_results, strict=True):
            result = {**old, **product_score(row, old['generated'] or '')}
            if old.get('skipped_context') or old.get('exception') or old.get('context_truncated'):
                for metric in SCORED:
                    result[metric] = False
            results.append(result)
        report['comparisons'].append(dict(path=str(path), report_sha256=digest(path),
            run=previous.get('run') or previous.get('model_id'), model_sha256=previous['model_sha256'],
            decoding=previous.get('decoding', 'grammar_instructed'),
            accounting_limit='Telemetry missing from the stored report is null, not assumed zero.',
            summary=summarize(rows, results)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({name: item['summary']['all'] for name, item in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
