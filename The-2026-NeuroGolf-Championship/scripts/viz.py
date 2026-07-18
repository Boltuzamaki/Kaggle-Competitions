import json

with open('c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task361.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    print(f'Train {i}')
    print('In:')
    for r in ex['input']: print(''.join(str(x) if x else '.' for x in r))
    print('Out:')
    for r in ex['output']: print(''.join(str(x) if x else '.' for x in r))
    print('-'*20)
