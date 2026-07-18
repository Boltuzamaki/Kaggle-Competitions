import json

d = json.load(open('data/task005.json'))
for i, ex in enumerate(d['train'] + d['test']):
    split = 'Train' if i < len(d['train']) else 'Test'
    idx = i if i < len(d['train']) else i - len(d['train'])
    print(f'\n--- {split} {idx} ---')
    print('IN:')
    for r in range(len(ex['input'])):
        row = ""
        for c in range(len(ex['input'][r])):
            val = ex['input'][r][c]
            row += str(val) if val != 0 else "."
        print(row)
    print('OUT:')
    for r in range(len(ex['output'])):
        row = ""
        for c in range(len(ex['output'][r])):
            val = ex['output'][r][c]
            row += str(val) if val != 0 else "."
        print(row)
