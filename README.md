# Product Category Classifier

This project predicts an e-commerce product category from the product title.

I built it around the information that would actually be available while a new product is being entered: its title. I also explored the other columns in the dataset, but I kept the final prediction pipeline title-based so the model can be used immediately, without waiting for views, merchant ratings, or other information that may not exist yet.

## Dataset

The original dataset contains **35,311 products**. During cleaning I removed rows without a product title or category, standardized a few category names that meant the same thing, and removed exact duplicate title/category pairs.

Examples of label cleanup:
- `Mobile Phone` → `Mobile Phones`
- `CPU` → `CPUs`
- `fridge` → `Fridges`

After cleaning and duplicate removal, **30,826 products** remained for modeling.

## Models I tried

I compared three text-classification approaches:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression + word TF-IDF | 0.9520 | 0.9543 |
| Linear SVC + word TF-IDF | 0.9603 | 0.9622 |
| Linear SVC + character TF-IDF | 0.9917 | 0.9918 |

The character-based Linear SVC performed best, with **99.17% accuracy** and **0.9918 macro F1** on the held-out test set.

Character n-grams worked especially well here because product titles contain model numbers, storage sizes, abbreviations and spelling variations. Pieces such as `iphone`, `kgv39`, `gb`, or parts of a camera model can still be useful even when the complete title has never appeared before.

## Project structure

```product-category-classifier/
├── models/
│   └── product_category_model.pkl
├── predict_category.py
├── product_category_classification.ipynb
├── products.csv
├── README.md
├── requirements.txt
└── train_model.py
```

## How to run the project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

Test the saved model interactively:

```bash
python predict_category.py
```

Then enter a product title, for example:

```text
bosch wap28390gb 8kg 1400 spin
```

The program returns:

```text
Predicted category: Washing Machines
```

## Manual check

I also tested the final model with the six product names supplied with the assignment. It classified all six as expected, including the less descriptive model-code examples such as `smeg sbs8004po`.

## Notes and possible improvements

The result is strong on this dataset, but I would still test the model on newer product data before treating the score as a production guarantee. Product names and model ranges change over time.

Another useful next step would be to inspect low-confidence or frequently confused categories and collect more examples from those areas. If the application later has reliable information such as brand or merchant data at prediction time, those features could also be tested separately.
