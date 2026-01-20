## 🍽️ Restaurant Review Sentiment Analyzer
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-blue?style=for-the-badge)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)

This project implements a web-based sentiment analyzer for restaurant reviews using a Flask backend. It primarily uses a **BERT (Bidirectional Encoder Representations from Transformers)** model for high-accuracy sentiment prediction, with a legacy **Naive Bayes** model available as a baseline. Users can enter a restaurant review into a text area, and the application will predict whether the sentiment of the review is "Positive" or "Negative".

## 📸 Preview
<table border="0">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/50add2c3-fd0a-49c5-913b-3158be824ffa" width="100%">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/cc4141a0-86f6-499a-a677-a48fb0b98067" width="100%">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/734231a2-afdd-43cd-b738-3c5a805e1f80" width="100%">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/f2eae6c6-c665-48b0-8aff-084126b2e16a" width="100%">
    </td>
  </tr>
</table>


## 🚀 Key Features

* **Interactive Web Interface**: User-friendly front-end for submitting reviews.
* **Real-time Sentiment Analysis**: Get instant sentiment predictions for your text.
* **Confidence Score**: Understand the model's certainty about its prediction.
* **Scalable Backend**: Built with Flask to handle review submissions and return results.

## 🛠️ Tech Stack
* Backend: Python, Flask
* Deep Learning: PyTorch, Transformers (Hugging Face)
* Machine Learning (Baseline): Scikit-Learn, Pandas, NumPy
* NLP: NLTK
* Frontend: HTML5, CSS3, JavaScript (AJAX)
 
## 📂 Project Structure

```
FlaskSentimentAnalyzer/
├── app.py
├── Restaurant_Reviews.tsv
├── sentiment_model_bert.pkl
├── model_bert/
├── restaurentreview_bert.ipynb
├── restaurentreview.ipynb (Baseline)
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    └── index.html
```
* `app.py`: The main Flask application that serves the web pages and handles sentiment prediction requests.
* `Restaurant_Reviews.tsv`: The dataset of restaurant reviews used for training and testing the sentiment model.
* `sentiment_model.pkl`: The pre-trained machine learning model pipeline.
* `.gitignore`: Specifies files and directories to be ignored by Git.
* `restaurentreview.ipynb`: Jupyter Notebook detailing the data preprocessing, model training, and evaluation steps.
* `static/`: Contains static files like CSS and JavaScript.
    * `style.css`: Stylesheet for the web application.
    * `script.js`: Frontend JavaScript for handling user input and communicating with the backend.
* `templates/`: Contains HTML template files.
    * `index.html`: The main HTML page for the sentiment analyzer.

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone [https://github.com/PamalaMadhumithaYadav/FlaskSentimentAnalyzer](https://github.com/PamalaMadhumithaYadav/FlaskSentimentAnalyzer)
cd FlaskSentimentAnalyzer
````

### 2\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### 3\. Activate the Virtual Environment

  * **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
  * **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4\. Install Dependencies

You'll need to install the required Python packages. If you don't have a `requirements.txt` file, you can install them manually:

```bash
pip install Flask torch transformers scikit-learn pandas nltk joblib
```

### 5\. Download NLTK Data

The project uses NLTK's stopwords. You need to download them once:

```python
# Open a Python interpreter or add this to a setup script
import nltk
nltk.download('stopwords')
```

### 6. Train the Model

#### Primary: BERT Model
If you need to retrain the BERT model, run the `restaurentreview_bert.ipynb` notebook.
1.  Open `restaurentreview_bert.ipynb` in Jupyter.
2.  Run all cells to fine-tune the BERT model and save the artifacts to `model_bert/` and `sentiment_model_bert.pkl`.

#### Baseline: Naive Bayes
To train the baseline Scikit-Learn model, run `restaurentreview.ipynb`.

### 7\. Run the Flask Application

```bash
python app.py
```

The application will start, and you can access it in your web browser at `http://127.0.0.1:5001/` (or `http://localhost:5001/`).

## 🕹️ Usage

1.  Open your web browser and navigate to `http://127.0.0.1:5001/`.
2.  Enter a restaurant review into the provided text area.
3.  Click the "Analyze Sentiment" button.
4.  The application will display the predicted sentiment (Positive or Negative) and a confidence score.

## 🤖 Model Details

### Primary Model: BERT
The application uses a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model for sentiment classification.
*   **Architecture**: `BertForSequenceClassification` from the Hugging Face Transformers library.
*   **Tokenizer**: `BertTokenizer` for preprocessing text.
*   **Training**: Fine-tuned on the `Restaurant_Reviews.tsv` dataset.

### Baseline Model: Naive Bayes (Legacy)
The project also includes a baseline model for comparison:
*   **TF-IDF Vectorizer**: `ngram_range=(1, 2)`, max 1500 features.
*   **Classifier**: Multinomial Naive Bayes.

### 📊 Evaluation Metrics

*(Note: These values are from the BERT model training.)*

  * **Accuracy**: 0.94
  * **Confusion Matrix**:
    ```
    [[96  0]
     [12 92]]
    ```
      * True Negatives: 96
      * False Positives: 0
      * False Negatives: 18
      * True Positives: 86
  * **Classification Report**:
    ```
                  precision    recall  f1-score   support

        Negative       0.89      1.00      0.94        96
        Positive       1.00      0.88      0.94       104

        accuracy                           0.94       200
       macro avg       0.94      0.94      0.94       200
    weighted avg       0.95      0.94      0.94       200
    ```

<!-- end list -->
