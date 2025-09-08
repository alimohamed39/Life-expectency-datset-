import pandas as pd

def create_data():
    return pd.read_csv("Life Expectancy Data.csv")



def drop_columns(data,print=False):
    na = data.isnull()
    data.dropna(inplace=True)
    data = data.drop(["Population", "Income composition of resources", "Total expenditure"], axis=1)
    if print:
        print("Data columns:", data.columns)

        print(f"The null values are ({na.sum()}):")

        print("New data shape", data.shape)
        data.dropna(inplace=True)
    else:
        pass

    return data



def describe(data):
    for col in data.columns:
        print(f"Data summary for col{col}:")
        print(data[col].describe())


def encode(data):
    data = pd.get_dummies(data,columns=["Country","Status"])
    return data






















