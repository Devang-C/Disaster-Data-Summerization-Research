# Text Summarization Algorithms for Turkey Disaster Data

This repository contains research work on text summarization algorithms applied to Turkey disaster data, including tweet classification using Bi-LSTM and various text summarization techniques.

## Table of Contents
1. [Installation](#installation)
2. [Tweet Classification with Bi-LSTM](#tweet-classification-with-bi-lstm)
3. [Text Summarization](#text-summarization)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Usage](#usage)

## Installation

To run the scripts in this repository:

1. Clone the repository:
   ```
   git clone 
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Tweet Classification with Bi-LSTM

We have implemented a Bi-LSTM model to classify tweets into various categories related to disaster response:

- Caution and advice
- Displaced people and evacuations
- Infrastructure and utility damage
- Injured or dead people
- Not humanitarian
- Other relevant information
- Requests or urgent needs
- Rescue, volunteering, or donation effort
- Sympathy and support

### Word2Vec Model

To use the Bi-LSTM classification, you need to download the pre-trained Word2Vec model:

1. Download the model from [Google News Vectors on Kaggle](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)
2. Create a models/ directory and Place the downloaded model file in it.

Note: The Word2Vec model is approximately 3.6 GB in size and is not included in this repository due to size constraints.

## Text Summarization

We have implemented various text summarization algorithms, including:

- ALBERT
- SBERT (Sentence-BERT)
- DistilBERT

## Evaluation Metrics

To assess the quality of the generated summaries, we use the following metrics:

- JS (Jensen-Shannon) Score
- ROUGE-L Score

## Usage

To run the summarization algorithms:

1. Execute the corresponding script for each algorithm
2. The script will generate a new CSV file containing:
   - Original text
   - Summarized text
   - JS Score
   - ROUGE-L Score


---
