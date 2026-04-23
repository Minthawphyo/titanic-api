import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os
load_dotenv()


SUPABASE_URL = "https://mrumlkaguhuejjponkrc.supabase.co"
SUPABASE_KEY = os.getenv("anon_key")

def seed():
    df = pd.read_csv("titanic.csv")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df             = df.dropna()
    df             = df.rename(columns={
        "Pclass": "pclass", "Sex": "sex", "Age": "age",
        "SibSp": "sibsp", "Parch": "parch", "Fare": "fare",
        "Embarked": "embarked", "Survived": "survived"
    })

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


    records = df.to_dict(orient="records")
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("passengers").insert(batch).execute()
        print(f"Inserted rows {i} to {i+len(batch)}")

    print(f"Seeded {len(df)} rows successfully")

if __name__ == "__main__":
    seed()