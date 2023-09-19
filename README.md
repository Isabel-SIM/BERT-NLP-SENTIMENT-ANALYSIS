# Text Classification with BERT

This repository demonstrates how to perform text classification using the BERT model with PyTorch and Hugging Face Transformers library. I will walk you through the process, including data loading, model training, evaluation, and inference.

---  

## Overview

- **NLP (Natural Language Processing):** Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and human language. It involves tasks like text classification, sentiment analysis, and text generation.

- **Sentiment Analysis:** Sentiment analysis is a subset of NLP that involves determining the sentiment or emotion expressed in a piece of text, such as positive, negative, or neutral.

- **BERT (Bidirectional Encoder Representations from Transformers):** BERT is a pre-trained language model that has achieved state-of-the-art results in various NLP tasks. It is designed to understand the context of words in a sentence by considering both the left and right context.

---  

## Training Details

- The BERT model was trained for text classification on the AG News dataset.
- The model was trained on 6 epochs.

---  

## Code Explanation

- **Data Loading:** I used the Hugging Face `datasets` library to load the AG News dataset and split it into training and testing sets.

- **Tokenization:** The text is tokenized using the BERT tokenizer, and I defined a maximum sequence length of 128 tokens.

- **Model:** Next, I loaded a pre-trained BERT model for sequence classification and set up the optimizer.

- **Training Loop:** The training loop runs for a specified number of epochs, calculating loss and updating model weights.

- **Evaluation:** I then, evaluated the trained model's accuracy on the test dataset.

- **Inference and Predictions:** The model is used for inference and predicting class labels, sentiment, and text generation.

---  

## Usage Instructions

To run this code, you'll need the following libraries:

```terminal
pip install transformers
pip install datasets
pip install torch torchvision torchaudio
pip install scikit-learn
```

---  

## Examples Generated via my BERT model..


1. **Sentiment Analysis:** Analyse the sentiment of a text.
    ```python
    text_to_analyze = "This is a fantastic product!"
    predicted_sentiment = "Positive"
    ```
    
2. **Text Classification:** Predict the class label of a given text.
    ```python
    text_to_classify = "The world cup has been fascinating."
    predicted_class = "Sports"
    ```
---  
   
## Examples Generated via a pre-trained model..


3. **Text Generation:** Generate text based on a prompt.
    ```python
    prompt = "Hello, world"
    generated_text = "Hello, world. I'm sorry, but I'm not sure what to do. I don't know what I should do, and I can't do anything. I want to be there for you, so I will be here. You'll see. It's not like I've been here for a long time. Maybe I was here before. Or maybe I just didn't want you to know. Either way, I think I need to..."
    ```

4. **Summarisation:** Summarise a long text using a BART model.
    ```python
    long_text = "Natural Language Processing (NLP) models are a subset of artificial intelligence (AI) that focuses on the interaction between computers and human language. These models aim to enable machines to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant. NLP has a wide range of applications, from sentiment analysis and text classification to machine translation and chatbots. One of the key breakthroughs in NLP is the development of transformer-based models, such as BERT and GPT, which have achieved remarkable results in various language understanding and generation tasks. These models have opened up new possibilities in language-related AI applications, making NLP a rapidly evolving field with exciting opportunities for research and innovation."
    generated_summary = "Natural Language Processing (NLP) models aim to enable machines to understand, interpret, and generate human language. NLP has a wide range of applications, from sentiment analysis and text classification to machine translation and chatbots. One of the key breakthroughs in NLP is the development of transformer-based models."

<br/>
    
---  

**Accuracy Achieved:** My BERT-based text classification model achieved an accuracy score of 94.38% on the test dataset.

---

This repository showcases the power of BERT-based models in NLP tasks, offering accurate text classification and sentiment analysis capabilities. Whether you're analysing sentiment in customer reviews, classifying news articles, or generating text, this code provides a strong foundation for future NLP research.
