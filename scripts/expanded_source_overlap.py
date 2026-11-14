"""Screen candidate source units against text-free reserved signature JSONL files.

``scan_units(root, units, manifest_paths)`` accepts units with ``unit_id``,
``family_id``, ``doc_id``, ``source_url``, and ``text``; an optional ``group`` is
also matched. Relative manifest paths resolve under ``root``. It returns file
SHA-256 pins, nonblank row counts, unique family IDs/counts, their union count,
and per-unit hits, partial-review evidence, and shared-8-gram totals. Manifest
paths in the result preserve the supplied spelling; ``signature_row`` is the
one-based physical JSONL line number, including any preceding blank lines.

Only candidate fingerprints are indexed in memory: a uint64 64-column MinHash
matrix and an inverted candidate word-8-gram index. Signature manifests are
hashed and parsed one row at a time, never indexed or loaded in their entirety.
Memory also includes one signature row, family-ID sets, and returned evidence.
The existing check_reserved_overlap fingerprint helpers are reused, without
calling its corpus/evaluation readers. No source text is returned or printed.

Hits are conservative screening signals, not adjudicated leakage: matching
IDs/families/groups, normalized URLs, normalized exact hashes, MinHash >= 0.5,
reserved-side 8-gram containment >= 0.5, or candidate-side containment >= 0.5
with at least 20 shared 8-grams. Rows with no hit but >= 50 shared 8-grams are
partial review. ``shared_ngram8_total`` sums intersections over all signature
rows, even below either threshold; repeated signatures count repeatedly, so
this is NOT a distinct shared-gram union. ``of_reserved`` and ``of_candidate``
are distinct 8-gram counts, not fractions. Empty 8-gram sets cannot establish
containment. MinHash is approximate and normalization inherits the existing
helpers. A zero-hit result is NOT clearance: coverage is limited to the pinned
manifests, signatures, source units, and these thresholds. Protected evaluation
questions/answers must not be supplied or opened by callers.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

if __package__:
    from .check_reserved_overlap import Signer, exact_hash, ngrams, url_key
else:
    from check_reserved_overlap import Signer, exact_hash, ngrams, url_key


def scan_units(root: Path, units: list[dict], manifest_paths: list[str]) -> dict:
    """Return text-free overlap evidence and manifest pins; never grant clearance.

    Unit IDs must be unique. Manifests use check_reserved_overlap's ``sign``
    schema, including 64 MinHash values and distinct hashed ``ngram8`` values.
    Blank lines are ignored for row counts but included in SHA-256 file pins.
    All supplied manifests are streamed, including when there are no units.
    """
    thresholds = {
        'minhash_estimate': 0.5,
        'ngram8_containment': 0.5,
        'candidate_ngram8_containment': 0.5,
        'candidate_min_shared_ngram8': 20,
        'partial_review_min_shared_ngram8': 50,
    }
    signer = Signer()
    signatures = np.empty((len(units), 64), dtype=np.uint64)
    equal_components = np.empty(signatures.shape, dtype=np.bool_)
    owners = defaultdict(list)
    by = {key: defaultdict(set) for key in ('id', 'family_id', 'group', 'url', 'exact')}
    candidate_sizes = []
    results = {}
    unit_ids = []
    for index, unit in enumerate(units):
        unit_id = unit['unit_id']
        if unit_id in results:
            raise ValueError('scan_units requires unique unit_id values')
        unit_ids.append(unit_id)
        results[unit_id] = {'hits': [], 'partial_review': [], 'shared_ngram8_total': 0}
        text = unicodedata.normalize('NFC', unit['text'])
        signatures[index] = signer(text)
        candidate_grams = ngrams(text)
        candidate_sizes.append(len(candidate_grams))
        for gram in candidate_grams:
            owners[gram].append(index)
        # Family IDs can also be collector document IDs or grouping keys.
        for value in (unit_id, unit.get('doc_id'), unit.get('family_id')):
            if value:
                by['id'][value].add(index)
        for value in (unit.get('family_id'), unit.get('group')):
            if value:
                by['group'][value].add(index)
        for key, value in (
            ('family_id', unit.get('family_id')),
            ('url', url_key(unit.get('source_url'))),
            ('exact', exact_hash(text)),
        ):
            if value:
                by[key][value].add(index)

    manifests = []
    union_families = set()
    for manifest_path in manifest_paths:
        digest = hashlib.sha256()
        families = set()
        row_count = 0
        with (root / manifest_path).open('rb') as source:
            for line_number, line in enumerate(source, 1):
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                row_count += 1
                family_id = row['family_id']
                families.add(family_id)
                hits = defaultdict(set)
                for key, value in (
                    ('id', row.get('doc_id')),
                    ('family_id', family_id),
                    ('group', row.get('group')),
                    ('url', url_key(row.get('source_url'))),
                    ('exact', row['exact_sha256']),
                ):
                    if value:
                        for index in by[key].get(value, ()):
                            hits[index].add(key)
                # A manifest family may identify a whole collector group or doc.
                if family_id:
                    for index in by['id'].get(family_id, ()):
                        hits[index].add('id')
                    for index in by['group'].get(family_id, ()):
                        hits[index].add('group')

                reserved_signature = np.asarray(row['minhash'], dtype=np.uint64)
                if reserved_signature.shape != (64,):
                    raise ValueError(
                        f'{manifest_path}:{line_number}: expected 64 MinHash values'
                    )
                np.equal(signatures, reserved_signature, out=equal_components)
                estimates = equal_components.mean(axis=1)
                for index in np.flatnonzero(estimates >= thresholds['minhash_estimate']):
                    hits[int(index)].add('minhash')

                reserved_grams = set(row['ngram8'])
                reserved_size = len(reserved_grams)
                shared = Counter(
                    index for gram in reserved_grams for index in owners.get(gram, ())
                )
                for index, count in shared.items():
                    results[unit_ids[index]]['shared_ngram8_total'] += count
                    if count / reserved_size >= thresholds['ngram8_containment']:
                        hits[index].add('containment')
                    if (
                        count >= thresholds['candidate_min_shared_ngram8']
                        and count / candidate_sizes[index]
                        >= thresholds['candidate_ngram8_containment']
                    ):
                        hits[index].add('candidate_containment')

                partial_indices = {
                    index for index, count in shared.items()
                    if count >= thresholds['partial_review_min_shared_ngram8']
                    and index not in hits
                }
                for index in sorted(hits.keys() | partial_indices):
                    methods = sorted(hits[index]) if index in hits else ['partial_ngram8']
                    entry = {
                        'manifest_path': manifest_path,
                        'signature_row': line_number,
                        'family_id': family_id,
                        'methods': methods,
                        'shared_ngram8': shared.get(index, 0),
                        'of_reserved': reserved_size,
                        'of_candidate': candidate_sizes[index],
                        'minhash_estimate': float(estimates[index]),
                    }
                    destination = 'hits' if index in hits else 'partial_review'
                    results[unit_ids[index]][destination].append(entry)
        union_families.update(families)
        manifests.append({
            'path': manifest_path,
            'sha256': digest.hexdigest(),
            'rows': row_count,
            'unique_family_ids': sorted(families),
            'unique_family_count': len(families),
        })

    return {
        'manifests': manifests,
        'union_signature_families': len(union_families),
        'units': results,
        'thresholds': thresholds,
    }
