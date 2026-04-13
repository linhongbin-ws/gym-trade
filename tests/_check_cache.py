from pathlib import Path
import pandas as pd

cache_dir = Path('.cache/1d')
need_start = pd.Timestamp('2022-02-18') - pd.DateOffset(years=1)
need_end   = pd.Timestamp('2022-02-18')
print(f'Need coverage: {need_start.date()} -> {need_end.date()}')

if not cache_dir.exists():
    print('Cache dir does not exist:', cache_dir)
else:
    files = list(cache_dir.glob('*_????-??-??_????-??-??.csv'))
    print(f'Total cached files: {len(files)}')
    covered, stale = [], []
    for f in files:
        parts = f.stem.rsplit('_', 2)
        try:
            first = pd.Timestamp(parts[1])
            last  = pd.Timestamp(parts[2])
            if first <= need_start and last >= need_end:
                covered.append(parts[0])
            else:
                stale.append((parts[0], str(first.date()), str(last.date())))
        except Exception:
            stale.append((f.stem, '?', '?'))
    print(f'Cover [{need_start.date()} ~ {need_end.date()}]: {len(covered)}')
    print(f'Stale / insufficient:                            {len(stale)}')
    if stale[:5]:
        print('Example stale:', stale[:5])
