# SMS Text Classification Project

## Overview
This project builds an SMS text classification model to distinguish between spam and non-spam (ham) messages. The model is trained using TF-IDF vectorization and a Linear Support Vector Classifier (LinearSVC).

## Features
- Loads and preprocesses the SMS Spam Collection dataset.
- Splits data into training and testing sets.
- Utilizes a machine learning pipeline with TF-IDF and LinearSVC.
- Provides a simple classification function for predicting new messages.
- Includes a Gradio interface for user-friendly interaction.

## Installation
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install pandas scikit-learn gradio
```

## Dataset
The dataset used is `SMSSpamCollection.csv`, which contains labeled messages.
### Expected Format:
| label | text_message |
|-------|-------------|
| ham   | Hello, how are you? |
| spam  | Win a free prize now! Click here. |

### Loading Data Correctly
Ensure the CSV is read properly using:
```python
import pandas as pd
sms_text_df = pd.read_csv("Resources/SMSSpamCollection.csv", sep="\t", names=["label", "text_message"])
```

## Model Training
The `sms_classification()` function:
1. Loads and preprocesses the dataset.
2. Splits data (67% training, 33% testing).
3. Uses a pipeline with `TfidfVectorizer()` and `LinearSVC(dual=False)`.
4. Returns the trained model and accuracy.

### Example Usage:
```python
model, accuracy = sms_classification(sms_text_df)
print("Model Accuracy:", accuracy)
```

## Gradio Interface
A simple UI is provided using Gradio for easy classification of new SMS messages.
To launch the interface:
```python
import gradio as gr
def predict_sms(message):
    return model.predict([message])[0]
interface = gr.Interface(fn=predict_sms, inputs="text", outputs="text")
interface.launch()
```

## Fixes Applied
- Corrected `pd.read_csv` separator (`\t`) and column names.
- Updated `LinearSVC(dual=False)` for compatibility.
- Modified `sms_classification()` to return accuracy.

## Authors
Team working on AI/ML projects.

## License
This project is open-source under the MIT License.

