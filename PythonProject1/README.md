# News Article Classification & Sentiment Analysis System

A containerized machine learning system for classifying news articles and performing sentiment analysis using modern NLP techniques.

## Features

- **FastAPI Backend**: RESTful API for classification and sentiment analysis
- **Streamlit Dashboard**: Interactive web interface
- **Transformer Models**: DistilBERT for classification and sentence transformers for embeddings
- **Vector Database**: ChromaDB for semantic search and article storage
- **Dockerized**: Complete containerized deployment with Docker Compose

## Project Structure

PythonProject1/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── database.py
│   └── utils.py
├── streamlit_ui/
│   └── app.py
├── docker/
│   ├── app.Dockerfile
│   └── streamlit.Dockerfile
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.20+
- 4GB+ RAM recommended

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project directory**

2. **Start the application**
   ```bash
   docker-compose up --build