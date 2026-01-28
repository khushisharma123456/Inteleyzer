import pandas as pd

xls = pd.ExcelFile('InteLeYzer_Database_20260128_031907.xlsx')
print('Sheets:', xls.sheet_names)
print()

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet, nrows=2)
    print(f'\n{sheet} columns:')
    print(df.columns.tolist())
    if len(df) > 0:
        print(f'Sample row: {df.iloc[0].to_dict()}')
