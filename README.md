# Heart Disease Prediction – Machine Learning Project 

A Python-based machine learning project to predict the presence of heart disease using supervised classification models. This project involves data preprocessing, model training using Logistic Regression and Random Forest, and performance evaluation with accuracy, confusion matrix, and classification reports.

## 🚀 Features

- 📊 Data cleaning and preprocessing from raw CSV
- 🔍 Logistic Regression and Random Forest Classifiers
- ✅ Model evaluation with:
  - Accuracy score
  - Confusion matrix
  - Classification report
- 📈 Visualizations using Seaborn and Matplotlib
- 📊 Accuracy comparison bar chart between models

## 📁 Files Included

```

Heart_Disease_ML/
├── data_cleaning.py            # Preprocesses dataset and creates features (X, y)
├── model_training.py           # Trains, evaluates, and compares ML models
├── heart.csv                   # Original dataset
├── heart_cleaned.csv           # Cleaned dataset used for model input
├── requirements.txt            # List of required Python libraries
├── README.md                   # This documentation

```

## 🧠 Models Used

- **Logistic Regression**
- **Random Forest Classifier**

## 🎯 Performance

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | **0.80** |
| Random Forest       | **0.97** |

📌 **Random Forest** outperformed Logistic Regression significantly in terms of both precision and recall.

## 🔎 Detailed Evaluation

### Logistic Regression

- **Accuracy**: 0.8019
- **Confusion Matrix**:
```

\[\[118  41]
\[ 20 129]]

```
- **Precision / Recall / F1**:
- Class 0: 86% precision, 74% recall
- Class 1: 76% precision, 87% recall

### Random Forest

- **Accuracy**: 0.9740
- **Confusion Matrix**:
```

\[\[157   2]
\[  6 143]]

````
- **Precision / Recall / F1**:
- Class 0: 96% precision, 99% recall
- Class 1: 99% precision, 96% recall

## 📈 Visualizations

- Confusion matrices for both models (Seaborn heatmaps)
- Bar chart comparing model accuracies

## Requirements 

Install dependencies using:

```bash
pip install -r requirements.txt
````

### requirements.txt includes:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 🛠 How to Run

1. Clone the repo
2. Make sure your terminal is in the project directory
3. Run preprocessing:

```bash
python data_cleaning.py
```
4. Run model training and evaluation:

```bash
python model_training.py
```

## 🧾 Dataset

The dataset used for this project was sourced from Kaggle – Heart Disease Dataset.
It contains medical attributes such as age, sex, chest pain type, cholesterol levels, and more, used to predict heart disease.(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). 

## 👤 Author

Laiba Maab 
Final-year Software Engineering Student | Developer | ML Learner
GitHub: [@laibamaab](https://github.com/laibamaab)

## ⭐️ Give It a Star!

If you found this project helpful, please ⭐ the repo! Feel free to fork, use, and extend it.
