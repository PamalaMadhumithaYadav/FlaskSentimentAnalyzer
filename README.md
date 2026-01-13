## 🍽️ Restaurant Review Sentiment Analyzer
This project implements a web-based sentiment analyzer for restaurant reviews using a Flask backend and a scikit-learn machine learning model. Users can enter a restaurant review into a text area, and the application will predict whether the sentiment of the review is "Positive" or "Negative". It also displays a confidence score for the prediction.

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
* Machine Learning: Scikit-Learn, Pandas, NumPy
* NLP: NLTK
* Frontend: HTML5, CSS3, JavaScript (AJAX)
 
## 📂 Project Structure

```
FlaskSentimentAnalyzer/
├── app.py
├── Restaurant\_Reviews.tsv
├── sentiment\_model.pkl
├── .gitignore
├── restaurentreview.ipynb
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
pip install Flask scikit-learn pandas nltk joblib
```

### 5\. Download NLTK Data

The project uses NLTK's stopwords. You need to download them once:

```python
# Open a Python interpreter or add this to a setup script
import nltk
nltk.download('stopwords')
```

### 6\. Train the Sentiment Model (if `sentiment_model.pkl` is not present or you want to retrain)

The `sentiment_model.pkl` file is provided, but if you need to retrain or understand the training process, run the `restaurentreview.ipynb` Jupyter Notebook.

  * Install Jupyter (if you don't have it):
    ```bash
    pip install jupyter
    ```
  * Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
  * Open `restaurentreview.ipynb` and run all cells. This will generate the `sentiment_model.pkl` file.

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

## 🤖 Model Details (from `restaurentreview.ipynb`)

The sentiment analysis model is a pipeline that consists of:

  * **TF-IDF Vectorizer**: Converts text data into numerical representations. It's configured to use a maximum of 1500 features and considers both unigrams and bigrams (`ngram_range=(1, 2)`).
  * **Multinomial Naive Bayes Classifier**: A probabilistic classifier suitable for text classification tasks.

The model was trained on the `Restaurant_Reviews.tsv` dataset.

### 📊 Evaluation Metrics (example from notebook output):

*(Note: These values are from a specific training run and may vary slightly upon re-training.)*

  * **Accuracy**: \~0.80
  * **Confusion Matrix**:
    ```
    [[83 13]
     [27 77]]
    ```
      * True Negatives: 83
      * False Positives: 13
      * False Negatives: 27
      * True Positives: 77
  * **Classification Report**:
    ```
                  precision    recall  f1-score   support

        Negative       0.75      0.86      0.81        96
        Positive       0.86      0.74      0.79       104

        accuracy                           0.80       200
       macro avg       0.81      0.80      0.80       200
    weighted avg       0.81      0.80      0.80       200
    ```

<!-- end list -->
