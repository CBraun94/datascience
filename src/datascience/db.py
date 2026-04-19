import polars as pl
from typing import Dict


OUT = r'/home/chris/dev/datascience/data/output'


class DataFrame():
    def __init__(self, name: str):
        self.name = name
        self.df: pl.DataFrame = None

    def write(self, path: str):
        if self.df is not None:
            _fpath = path + '/dataframe.xlsx'
            self.df.write_excel(workbook=_fpath, worksheet='DataFrame')

    def read(self, path: str):
        import os
        _fpath = path + '/dataframe.xlsx'
        if os.path.exists(_fpath):
            self.df = pl.read_excel(source=path + '/dataframe.xlsx', sheet_name='DataFrame')


class Database():
    def __init__(self):
        self.data: Dict[str, DataFrame] = {}


def get_test_df():
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.DataFrame(data)
    return df


def main():
    print('main')
    df = DataFrame('test')

    df.read(OUT)

    #df.df = get_test_df()

    print(df.df)

    df.write(OUT)


if __name__ == '__main__':
    main()
