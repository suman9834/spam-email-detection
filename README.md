# 📧 Spam/Ham Email Classification using NLP & LSTM

A Natural Language Processing and Deep Learning project that classifies
email messages as **Spam** or **Ham (Not Spam)** using text
preprocessing, tokenization, sequence padding, and an **LSTM neural
network** built with TensorFlow/Keras.

## 📌 Project Overview

This project uses the `spam_ham_dataset.csv` dataset and follows an
end-to-end text classification workflow:

-   Dataset loading and exploration
-   Missing-value and duplicate checking
-   Spam/Ham distribution analysis
-   Class balancing by downsampling the Ham class
-   Text preprocessing
-   Punctuation removal
-   Stopword removal
-   WordCloud visualization
-   Train/test splitting
-   Text tokenization
-   Sequence padding
-   LSTM model building
-   Model training with Early Stopping and ReduceLROnPlateau
-   Model evaluation
-   Confusion Matrix
-   Classification Report
-   Accuracy and Loss visualization
-   Prediction on new messages

## 🗂️ Dataset

The project uses:

``` text
spam_ham_dataset.csv
```

The original dataset contains **5,171 records** and 4 columns:

  Column         Description
  -------------- -----------------------------------
  `Unnamed: 0`   Original dataset/index identifier
  `label`        Email class: `ham` or `spam`
  `text`         Email/message text
  `label_num`    Numeric label

### Original Class Distribution

-   Ham: **3,672 (71.01%)**
-   Spam: **1,499 (28.99%)**

To reduce class imbalance, the project downsamples Ham messages to the
number of Spam messages, producing a balanced dataset of **2,998
messages**.

## 🔄 Workflow

``` text
Dataset
   ↓
Data Exploration
   ↓
Missing & Duplicate Check
   ↓
Spam/Ham Distribution
   ↓
Class Balancing
   ↓
Text Cleaning
   ↓
Punctuation Removal
   ↓
Stopword Removal
   ↓
WordCloud
   ↓
Train/Test Split
   ↓
Tokenization
   ↓
Padding
   ↓
LSTM Model
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
   ↓
Confusion Matrix & Classification Report
   ↓
New Message Testing
```

## 🧹 Text Preprocessing

The notebook performs the following preprocessing steps:

1.  Removes the `Subject` text from messages.
2.  Removes punctuation.
3.  Converts words to lowercase during stopword processing.
4.  Removes English stopwords.
5.  Converts cleaned text into numerical sequences using a Keras
    tokenizer.
6.  Pads/truncates sequences to a maximum length of **100 tokens**.

## 🧠 LSTM Model

The model contains:

-   Embedding layer
-   LSTM layer with 16 units
-   Dense layer with 32 ReLU units
-   Dense output layer with sigmoid activation

The model is compiled using:

-   **Loss:** Binary Crossentropy
-   **Optimizer:** Adam
-   **Metric:** Accuracy

## ⚙️ Training

The model is trained for up to **20 epochs** with:

-   Batch size: **32**
-   Early Stopping
-   ReduceLROnPlateau
-   80/20 train-test split

Early Stopping restores the best model weights based on validation
accuracy.

## 📊 Evaluation

The project evaluates the model using:

-   Test Loss
-   Test Accuracy
-   Confusion Matrix
-   Precision
-   Recall
-   F1-score
-   Accuracy curve
-   Loss curve

### Current Notebook Result

The saved notebook currently reports:

-   **Test Accuracy: \~55.33%**
-   **Test Loss: \~0.6924**

Classification report:

  Class     Precision   Recall   F1-Score
  ------- ----------- -------- ----------
  Ham            0.55     0.87       0.67
  Spam           0.59     0.20       0.30

This result represents the current implementation and provides a useful
baseline for further model improvement.

## 🧪 New Message Prediction

The notebook also includes a prediction function that accepts a new
message and classifies it as:

``` text
Spam
```

or

``` text
Ham
```

Example test messages included in the notebook cover both
promotional/spam-like and normal conversational text.

## 🛠️ Technologies Used

-   **Python**
-   **Jupyter Notebook**
-   **Pandas**
-   **NumPy**
-   **Matplotlib**
-   **Seaborn**
-   **NLTK**
-   **WordCloud**
-   **Scikit-learn**
-   **TensorFlow**
-   **Keras**
-   **LSTM**

## 📁 Project Structure

``` text
Spam-Email-Detection/
│
├── spam_ham_dataset.csv
├── spam_Email_dectec.ipynb.ipynb
└── README.md
```

> You can rename the notebook to something more professional, such as
> `spam_ham_lstm.ipynb`, before publishing the repository.

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/suman9834/spam-email-detection.git
cd spam-email-detection
```

### 2. Install the required libraries

``` bash
pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn tensorflow
```

### 3. Start Jupyter Notebook

``` bash
jupyter notebook
```

### 4. Open the notebook

Open:

``` text
spam_Email_dectec.ipynb
```

Make sure `spam_ham_dataset.csv` is in the same directory as the
notebook.

### 5. Run the cells

Run the notebook from top to bottom.

## 🔮 Future Improvements

The current model provides a baseline, but its performance can be
improved by:

-   Using a separate validation set instead of using the test set for
    validation
-   Better text normalization
-   Trying Bidirectional LSTM
-   Using GRU
-   Increasing model capacity carefully
-   Hyperparameter tuning
-   Using pretrained word embeddings
-   Trying Transformer-based text classification
-   Comparing the LSTM with traditional ML models such as TF-IDF +
    Logistic Regression or SVM
-   Improving the handling of email-specific features

## 🎯 Learning Outcomes

This project helped demonstrate practical understanding of:

-   Exploratory Data Analysis
-   NLP preprocessing
-   Text tokenization
-   Sequence padding
-   Class balancing
-   Deep Learning
-   LSTM networks
-   Model evaluation
-   Binary text classification
-   Real-world message prediction

## 👨‍💻 Author

**Suman Kumar**

B.Tech -- Computer Science & Engineering\
Specialization: Artificial Intelligence & Data Science

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star!
