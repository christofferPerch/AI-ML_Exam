import pandas as pd

df = pd.read_csv("../../data/raw/heart_2022_with_nans.csv")

# Dropping all the columns, that we think might not influence the chance of a Heart Disease:
df.drop(
    columns=[
        "State",
        "LastCheckupTime",
        "RemovedTeeth",
        "HadSkinCancer",
        "HadCOPD",
        "HadArthritis",
        "DeafOrHardOfHearing",
        "BlindOrVisionDifficulty",
        "DifficultyConcentrating",
        "DifficultyDressingBathing",
        "DifficultyErrands",
        "ChestScan",
        "FluVaxLast12",
        "PneumoVaxEver",
        "TetanusLast10Tdap",
        "HighRiskLastYear",
        "CovidPos",
    ],
    inplace=True,
)

df.to_csv("../../data/processed/heart_2022.csv", index=False)
