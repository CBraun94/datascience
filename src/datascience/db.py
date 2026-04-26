import polars as pl
from typing import Dict


OUT = r'/home/chris/dev/datascience/data/output'
DF_XLSX = '/dataframe.xlsx'
DF = 'DataFrame'


class DataFrame():
    def __init__(self, name: str):
        self.name = name
        self.df: pl.DataFrame = None
        self.data: Dict[str] = {}
        self.config: Dict[str] = {}

    @staticmethod
    def read_dir(path: str):
        import os
        r = {}
        d = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        for i in d:
            df = DataFrame(name=i)
            df.read(path=path + os.pathsep + i)
            r[i] = df

        return r

    def write(self, path: str):
        self.__write_df(path=path)
        self.__write_data(path=path)
        self.__write_config(path=path)

    def read(self, path: str):
        self.__read_df(path=path)
        self.__read_data(path=path)
        self.__read_config(path=path)

    def __write_df(self, path: str):
        if self.df is not None:
            _fpath = path + DF_XLSX
            self.df.write_excel(workbook=_fpath, worksheet=DF)

    def __write_data(self, path: str):
        import json
        with open(file=path+'/data.json', mode='w') as f:
            f.write(json.dumps(self.data))

    def __write_config(self, path: str):
        import json
        with open(file=path+'/config.json', mode='w') as f:
            f.write(json.dumps(self.config))

    def __read_df(self, path: str):
        import os
        _fpath = path + DF_XLSX
        if os.path.exists(_fpath):
            self.df = pl.read_excel(source=path + DF_XLSX, sheet_name=DF)

    def __read_data(self, path: str):
        import json
        import os
        _fpath = path+'/data.json'
        if os.path.exists(_fpath):
            with open(file=_fpath, mode='r') as f:
                self.data = json.loads(f.read())

    def __read_config(self, path: str):
        import json
        import os
        _fpath = path+'/config.json'
        if os.path.exists(_fpath):
            with open(file=_fpath, mode='r') as f:
                self.config = json.loads(f.read())


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
