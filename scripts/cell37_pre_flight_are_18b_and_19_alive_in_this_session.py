# ===== PRE-FLIGHT - are 18b and 19 alive in this session? =====
try:
    m, c = build('optuna_best')
    ds_ok = 'VDEIDatasetV2' in globals()
    print(f'CELL 19 OK (params={sum(p.numel() for p in m.parameters()):,})  |  CELL 18b OK (dataset={ds_ok})')
    print('-> ready for Cell B ✅')
except NameError as e:
    print(f'MISSING: {e} -> run CELL 18b / CELL 19 first')
