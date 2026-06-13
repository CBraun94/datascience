from . import db
from . import ds


def run():
    r = db.DataFrame.read_dir(db.OUT)
    print(r)
    #ds.main()


if __name__ == '__main__':
    run()
