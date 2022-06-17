from BiLSTM import Train
from Test import Test
from CSVWrite import CSVWrite
from CreateDataset import CreateData
import sys

args = sys.argv[1]
if args == "BiLSTM":
    Train().main()
elif args == "CreateDataset":
    CreateData().main()
elif args == "Test":
    Test().main()
elif args == "CSVWrite":
    CSVWrite().main()
else:
    print("実行ファイルがありません")