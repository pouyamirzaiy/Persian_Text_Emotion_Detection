# **Persian Text Emotion Detection**

## **Project Overview**

This project involves performing emotion detection on a collection of Persian texts categorized into five emotional classes: happiness, sadness, anger, fear, and others. The project includes data cleaning, feature engineering, model training, evaluation, and inference.

## **Libraries and Dependencies**

The following libraries are used in this project:

- **Pandas**: Data processing
- **Scikit-learn**: Preprocessing, model training, and evaluation
- **Hazm**: Persian text processing
- **NLTK**: Natural language processing
- **Matplotlib**: Visualization

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

```

## **Project Structure**

- `src/`
  - `data_cleaning.py`: Script for data cleaning and preprocessing.
  - `feature_engineering.py`: Script for feature engineering.
  - `model_training.py`: Script for training and evaluating models.
  - `grid_search.py`: Script for performing grid search to find the best hyperparameters.
  - `inference.py`: Script for making predictions on the test set using the trained model.
- `notebooks/`
  - `Emotion.ipynb`: Jupyter notebook for exploratory data analysis and model evaluation.
- `data/`: Contains the training and test datasets.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required libraries and their versions.

## **Data Cleaning**

The data cleaning phase includes handling missing values and transforming text data for better analysis. Detailed steps can be found in the `data_cleaning.py` script.

## **Feature Engineering**

Feature engineering involves vectorizing text data using TF-IDF and creating new features. Detailed steps can be found in the `feature_engineering.py` script.

## **Model Training and Evaluation**

Various machine learning models are trained and evaluated using different metrics. The models include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- XGBoost Classifier

Detailed steps can be found in the `model_training.py` script.

## **Grid Search**

Grid search is performed to find the best hyperparameters for the Random Forest model. Detailed steps can be found in the `grid_search.py` script.

## **Inference**

The inference process involves making predictions on the test set using the trained model. Detailed steps can be found in the `inference.py` script.

## **Contact Information**

If you have any questions or feedback, feel free to reach out to me at pouya.8226@gmail.com
```
