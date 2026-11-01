"""MODEL-RESUME-A-025: print what differs between code versions in reproduce_routing.py output.
usage: summarize_routing.py <routing.json> <label> [<label> ...]   (all turns listed; '*' marks a difference)"""
import json
import sys

d = json.load(open(sys.argv[1]))
labels = sys.argv[2:]
sides = [d['sides'][k] for k in labels]


def s(r):
    x = r['status'] + (f"/{r['reason']}" if r.get('reason') else '') + (f" [{r['index']}]" if r['index'] != 'v5' else '')
    if r.get('applied'):
        x += ' → ' + r['applied']
    if r.get('evidence_tables'):
        x += f" 별표{','.join(r['evidence_tables'])}"
    if r.get('options'):
        x += ' → ' + '; '.join(o[:30] for o in r['options'])
    if r.get('table_condition'):
        x += f" tc={r['table_condition']}"
    if r.get('tables_set_aside'):
        x += f" aside={r['tables_set_aside']}"
    if r.get('class_notice'):
        x += ' [class notice]'
    return x


changed = 0
for q in sides[0]['single']:
    rows = [s(side['single'][q]) for side in sides]
    diff = len(set(rows)) > 1
    changed += diff
    print(('* ' if diff else '  ') + q)
    for label, row in zip(labels, rows):
        print(f'     {label:10} {row}')
print(f"single questions differing across {', '.join(labels)}: {changed} of {len(sides[0]['single'])}")
for name in sides[0]['conversations']:
    print('==', name)
    for turns in zip(*(side['conversations'][name] for side in sides)):
        rows = [s(t) for t in turns]
        print(('* ' if len(set(rows)) > 1 else '  ') + turns[0]['q'])
        for label, row in zip(labels, rows):
            print(f'     {label:10} {row}')
