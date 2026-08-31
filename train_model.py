from pathlib import Path
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "products.csv"
MODEL_PATH = BASE_DIR / "models" / "product_category_model.pkl"

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Product Title", "Category Label"]).copy()
    df["Product Title"] = df["Product Title"].astype(str).str.strip().str.lower()
    df["Category Label"] = df["Category Label"].astype(str).str.strip()

    # A few labels describe the same category with different wording.
    df["Category Label"] = df["Category Label"].replace({
        "fridge": "Fridges",
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
    })

    df = df[(df["Product Title"] != "") & (df["Category Label"] != "")]
    df = df.drop_duplicates(subset=["Product Title", "Category Label"])
    return df

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=120000
        )),
        ("model", LinearSVC(random_state=42))
    ])

def main():
    df = load_and_clean_data(DATA_PATH)
    model = build_model()
    model.fit(df["Product Title"], df["Category Label"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Model trained on {len(df):,} products.")
    print(f"Saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
