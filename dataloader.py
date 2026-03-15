import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

header = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

numeric_cols = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

categorical_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class AdultDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.folder = 'adult/'

        df = self._load_data(split)

        self.category_maps = self._build_category_maps()

        # Encode categorical columns
        for col in categorical_cols:
            mapping = self.category_maps[col]
            df[col] = df[col].map(mapping)

            # In case unseen categories appear
            if df[col].isna().any():
                raise ValueError(f"Unseen category found in column '{col}'")

        # Encode target
        target_map = {"<=50K": 0, ">50K": 1}
        df["income"] = df["income"].map(target_map)

        if df["income"].isna().any():
            raise ValueError("Unexpected value found in 'income' column")

        self.df = df

        self.x_num = torch.tensor(
            self.df[numeric_cols].values, dtype=torch.float32
        )
        self.x_cat = torch.tensor(
            self.df[categorical_cols].values, dtype=torch.long
        )
        self.y = torch.tensor(
            self.df["income"].values, dtype=torch.float32
        )

    def _load_data(self, split):
        if split == "train":
            path = self.folder + "adult.data"
            df = pd.read_csv(path, names=header)

        elif split == "test":
            path = self.folder + "adult.test"
            df = pd.read_csv(path, names=header, skiprows=1)

        elif split == "merge":
            train_df = pd.read_csv(self.folder + "adult.data", names=header)
            test_df = pd.read_csv(self.folder + "adult.test", names=header, skiprows=1)

            df = pd.concat([train_df, test_df], ignore_index=True)

        else:
            raise ValueError("split must be 'train' or 'test'")

        # Remove whitespace around strings
        for col in df.select_dtypes(include=["object", "str"]).columns:
            df[col] = df[col].str.strip()

        df = df.replace("?", np.nan).dropna()

        return df

    def _build_category_maps(self):
        full_df = self._load_data("merge")

        category_maps = {}
        for col in categorical_cols:
            categories = sorted(full_df[col].unique())
            category_maps[col] = {cat: idx for idx, cat in enumerate(categories)}

        return category_maps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_num = self.x_num[idx]
        x_cat = self.x_cat[idx]
        y = self.y[idx]

        if self.transform:
            x_num, x_cat, y = self.transform(x_num, x_cat, y)

        return x_num, x_cat, y

if __name__ == "__main__":
    train_dataset = AdultDataset(split="train")

    x_num, x_cat, y = train_dataset[0]

    print("Numeric features:", x_num)
    print("Categorical features:", x_cat)
    print("Label:", y)

    from model import AdultNet

    cat_cardinalities = []
    for col in categorical_cols:
        cat_cardinalities.append(train_dataset.df[col].nunique())

    model = AdultNet(
        num_numeric=len(numeric_cols),
        cat_cardinalities=cat_cardinalities
    )

    print(model.mlp)