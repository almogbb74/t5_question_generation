# T5-Powered Question Generation Service
## Overview

This repository contains a stateless Python/Flask RESTful API designed for dynamic question generation.
The service utilizes a sophisticated Natural Language Processing (NLP) pipeline to consume public-domain text (via Web Scraping), extract key concepts using TF-IDF, and feed the results into a T5 (Text-to-Text Transfer Transformer) model to programmatically generate quiz questions.

This API serves as the intelligent backend for mobile or web applications, providing a scalable and efficient method for creating contextual and engaging learning content on demand.

## 💡 NLP Pipeline Highlights

The API executes a three-stage NLP pipeline upon receiving a subject query:

Dynamic Content Sourcing: A dedicated function performs Web Scraping (specifically targeting Wikipedia) to retrieve and clean the introductory text relevant to the user's requested subject.

Intelligent Keyword Extraction (TF-IDF): The scraped text is processed using Term Frequency-Inverse Document Frequency (TF-IDF) techniques. This algorithm statistically scores the importance of each non-stop word within the document, ensuring the most relevant concepts are selected as the "answers" for the generated questions.

Contextual Question Generation: The T5 Sequence-to-Sequence (Seq2Seq) model from the Hugging Face transformers library is loaded using PyTorch. The full text and the top TF-IDF keywords are formatted as "context: [text] answer: [keyword]" and passed to the model. Beam Search is used during inference to optimize the quality and coherence of the resulting questions.
