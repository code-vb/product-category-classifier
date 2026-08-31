from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "product_category_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

def main():
    model = load_model()
    print("Product Category Predictor")
    print("Type a product title to get a category suggestion.")
    print("Type 'quit' to stop.\n")

    while True:
        title = input("Product title: ").strip()

        if title.lower() in {"quit", "exit", "q"}:
            print("Done.")
            break

        if not title:
            print("Please enter a product title.\n")
            continue

        category = model.predict([title])[0]
        print(f"Predicted category: {category}\n")

if __name__ == "__main__":
    main()
