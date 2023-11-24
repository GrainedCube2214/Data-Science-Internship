datasets = ["NIFTY 50-01-01-2015-to-31-12-2015.csv", "NIFTY 50-01-01-2016-to-31-12-2016.csv", "NIFTY 50-01-01-2017-to-31-12-2017.csv", 
            "NIFTY 50-01-01-2018-to-31-12-2018.csv", "NIFTY 50-01-01-2019-to-31-12-2019.csv", "NIFTY 50-01-01-2020-to-31-12-2020.csv", 
            "NIFTY 50-01-01-2021-to-31-12-2021.csv", "NIFTY 50-01-01-2022-to-31-12-2022.csv", "NIFTY 50-31-10-2022-to-31-10-2023.csv"]

path = "D:/Downloads/The Interregnum Semester/Internship/StockMarketPred-SrcData/"
files = [path + dataset for dataset in datasets]
dpath = "D:/Downloads/The Interregnum Semester/Internship/StockMarketData2015-2023.csv"
dest = open(dpath, "w")

for i in range(len(files)):
    f = open(files[i], "r")
    if i == 0:
        dest.write(f.read())
    else:
        f.readline()
        dest.write(f.read())
    dest.write("\n")
    f.close()