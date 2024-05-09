import pandas as pd


df = pd.read_csv("../data/heart_2022_with_nans.csv")

pd.set_option('display.max_columns', None)

df.head()

df.info()

df.drop(columns=["State", "LastCheckupTime", "RemovedTeeth", "HadSkinCancer", "HadCOPD", 
                 "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty", 
                 "DifficultyConcentrating", "DifficultyDressingBathing", "DifficultyErrands", 
                 "ChestScan", "FluVaxLast12", "PneumoVaxEver", 
                 "TetanusLast10Tdap", "HighRiskLastYear", "CovidPos"], inplace=True)

df.head()

df.info()

df.to_csv("../data/heart_2022_removed_columns.csv",index=False)