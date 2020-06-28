from typing import List
import pandas as pd

data = pd.read_csv("data/Rumelhart_livingthings.csv", sep=",")

columns = [
    "Grow",
    "Living",
    "LivingThing",
    "Animal",
    "Move",
    "Skin",
    "Bird",
    "Feathers",
    "Fly",
    "Wings",
    "Fish",
    "Gills",
    "Scales",
    "Swim",
    "Yellow",
    "Red",
    "Sing",
    "Robin",
    "Canary",
    "Sunfish",
    "Salmon",
    "Daisy",
    "Rose",
    "Oak",
    "Pine",
    "Green",
    "Bark",
    "Big",
    "Tree",
    "Branches",
    "Pretty",
    "Petals",
    "Flower",
    "Leaves",
    "Roots",
    "Plant",
]
index = ["Robin", "Canary", "Sunfish", "Salmon", "Daisy", "Rose", "Oak", "Pine"]

# Drop some columns & indices to simplify our dataset
df = (
    pd.pivot_table(
        data, values="TRUE", index=["Item"], columns=["Attribute"], fill_value=0
    )
    .astype(float)
    .reindex(index, axis="index",)
    .reindex(columns, axis="columns",)
)

# In the 2020 paper, roses have no leaves; only petals (relevant for orthogonality)
df["Leaves"]["Rose"] = 0.0
# print(df["Grow"])

# Let's use a subset of the Rumelhart set for simplicity
df_limited = (
    df.drop(index=["Daisy", "Pine", "Robin", "Sunfish"])
    .drop(
        columns=list(
            filter(
                lambda x: x
                not in ["Grow", "Move", "Roots", "Fly", "Swim", "Leaves", "Petals"],
                columns,
            )
        )
    )
    .reindex(["Canary", "Salmon", "Oak", "Rose"], axis="index")
    .reindex(["Grow", "Move", "Roots", "Fly", "Swim", "Leaves", "Petals"], axis="columns")
)

# df_to_use = df
df_to_use = df_limited

items = sorted(df_to_use.index.unique())
attributes = sorted(df_to_use.columns.unique())


def get_data() -> [List, List, pd.DataFrame]:
    return [items, attributes, df_to_use]
