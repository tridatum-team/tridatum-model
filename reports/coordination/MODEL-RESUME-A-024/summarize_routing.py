import json
import sys

d = json.load(open(sys.argv[1]))


def s(r):
    x = r['status'] + (f" [{r['index']}]" if r['index'] != 'v5' else '')
    if r.get('applied'):
        x += ' → ' + r['applied'] + (f" ({r.get('scope')})" if r.get('scope') else '')
    if r.get('linked') or r.get('rows_annex'):
        x += f" linked={r.get('linked')} rows=별표{r.get('rows_annex')}"
    if r.get('options'):
        x += ' → ' + '; '.join(o[:34] for o in r['options'][:4])
    if r.get('table_condition'):
        x += f" tc={r['table_condition']}"
    if r.get('class_notice'):
        x += ' [notice]'
    return x


o, c = d['operating'], d['candidate']
n = 0
for q in o['single']:
    a, b = s(o['single'][q]), s(c['single'][q])
    if a != b:
        n += 1
        print('*', q); print('   op :', a); print('   new:', b)
print('single changed', n, 'of', len(o['single']))
for name in o['conversations']:
    print('==', name)
    for x, y in zip(o['conversations'][name], c['conversations'][name]):
        mark = '*' if s(x) != s(y) else ' '
        print(f" {mark} {x['q']}\n     op : {s(x)}\n     new: {s(y)}")
print('domain op:', o['domain'])
print('domain new:', {k: v for k, v in (c['domain'] or {}).items() if k != 'term_scope'})
