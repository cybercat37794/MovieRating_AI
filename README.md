# MovieRating_AI

This repository contains the code and resources for the MovieRating_AI project, which aims to predict movie ratings based on their plot summaries using transformer models.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The MovieRating_AI project leverages natural language processing (NLP) techniques to predict movie ratings (Good, Average, Bad) based on their plot summaries. The project uses transformer models such as BERT and RoBERTa to achieve this task.

## Dataset

The dataset used in this project contains movie plot summaries and their corresponding ratings. The ratings are categorized into three classes: Good, Average, and Bad. The dataset is preprocessed to handle missing values and duplicates.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. **Preprocess the Data:** Run the Movie_Data_Prep.ipynb notebook to preprocess the dataset. This step includes cleaning the data, handling missing values, and removing duplicates.

2. **Train the Model:** Use the movie_rating_model2_BERD.py script to train the transformer model on the preprocessed dataset. The script includes data tokenization, model building, training, and evaluation.

3. **Evaluate the Model:** The script will output the accuracy, precision, recall, and F1 score of the model on the test set.

## Model
The project uses transformer models such as BERT and RoBERTa for predicting movie ratings. The models are fine-tuned on the movie plot summaries to classify the ratings into three categories: Good, Average, and Bad.

### Model Architecture
* **Input Layer:** Tokenized movie plot summaries.
* **Transformer Layer:** Pre-trained transformer model (BERT or RoBERTa).
* **Dense Layers:** Fully connected layers with ReLU activation.
* **Output Layer:** Softmax activation for multi-class classification.

### Training
* **Optimizer:** AdamW
* **Loss Function:** Sparse Categorical Crossentropy
* **Metrics:** Accuracy, Precision, Recall, F1 Score

### Results
The model's performance is evaluated using accuracy, precision, recall, and F1 score. The results are printed after the training process.

### Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.