# 📩 Email Spam Classifier

## 📌 Project Overview

The Email Spam Classifier is a Machine Learning project that classifies text messages into two categories: **Spam** and **Legitimate (Ham)**.

This project uses Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm to identify unwanted messages based on their text content.

## 🎯 Objectives

- Identify spam and legitimate messages.
- Apply Natural Language Processing techniques.
- Convert text data into numerical features using TF-IDF.
- Train a machine learning classification model.
- Build a simple web application using Streamlit.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data handling and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning
- **NLP** – Text processing
- **TF-IDF Vectorizer** – Feature extraction
- **Multinomial Naive Bayes** – Classification algorithm
- **Streamlit** – Web application interface

## 📂 Dataset

**SMS Spam Collection Dataset**

The dataset contains SMS messages labelled as:

- `ham` – Legitimate messages
- `spam` – Unwanted messages

Dataset source: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## ⚙️ Project Workflow

1. Load the SMS dataset.
2. Perform basic data cleaning.
3. Separate messages and labels.
4. Split the dataset into training and testing sets.
5. Convert text into numerical features using TF-IDF.
6. Train the Multinomial Naive Bayes model.
7. Evaluate the model using accuracy, precision, recall and confusion matrix.
8. Save the trained model.
9. Use Streamlit to classify new messages.

## 📁 Project Structure

```text
Email-Spam-Classifier/
│
├── dataset/
│   └── spam.csv
│
├── app.py
├── train_model.py
├── requirements.txt
├── spam_model.pkl
├── vectorizer.pkl
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Richal14/Email-Spam-Classifier.git
```

### 2. Open the project folder

```bash
cd Email-Spam-Classifier
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the machine learning model

```bash
python train_model.py
```

### 7. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🚀 Future Enhancements

- Improve classification performance.
- Add a more interactive user interface.
- Support email text classification.
- Deploy the application online.

## How it looks?
<img width="1907" height="951" alt="Screenshot 2026-11-09 192334" src="https://github.com/user-attachments/assets/a7c43220-426f-486f-854c-6f2e87919bf4" />

Local URL: http://localhost:8503(Search to check)

## 👩‍💻 Author

**Richal**

GitHub: [Richal14](https://github.com/Richal14)

## ⭐ Acknowledgement

Thanks to the creators of the SMS Spam Collection dataset for providing data for this machine learning project.
