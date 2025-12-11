import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing pipeline for news articles"""

    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass

        self.stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.data.find('corpora') else set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, text: str) -> str:
        """Full preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned) if 'punkt' in nltk.data.find('tokenizers') else cleaned.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)


class SimpleEmbeddingModel:
    """Simple embedding model without external dependencies"""

    def __init__(self, vocab_size=10000, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.embeddings = None
        self._build_vocab()

    def _build_vocab(self):
        """Build a simple vocabulary from common words"""
        common_words = [
            'the', 'and', 'that', 'for', 'with', 'this', 'from', 'have', 'was', 'are',
            'news', 'article', 'report', 'said', 'government', 'company', 'market',
            'business', 'technology', 'science', 'health', 'sports', 'entertainment',
            'stock', 'price', 'market', 'economy', 'political', 'world', 'national'
        ]

        for i, word in enumerate(common_words):
            self.word_to_idx[word] = i

        # Initialize random embeddings
        np.random.seed(42)
        self.embeddings = np.random.randn(len(self.word_to_idx), self.embedding_dim)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding (average of word vectors)"""
        words = text.lower().split()
        word_vectors = []

        for word in words:
            if word in self.word_to_idx:
                word_vectors.append(self.embeddings[self.word_to_idx[word]])

        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)


class LocalNewsClassifier:
    """Local news classifier without Hugging Face"""

    def __init__(self):
        logger.info("Initializing Local News Classifier")
        self.preprocessor = TextPreprocessor()
        self.embedding_model = SimpleEmbeddingModel()

        # News categories
        self.categories = [
            "politics", "business", "technology", "science",
            "health", "sports", "entertainment", "world"
        ]

        # Simple rule-based classifier
        self.category_keywords = {
            "politics": ['government', 'election', 'president', 'minister', 'vote', 'policy'],
            "business": ['market', 'stock', 'economy', 'company', 'business', 'profit'],
            "technology": ['tech', 'computer', 'software', 'digital', 'internet', 'ai'],
            "science": ['science', 'research', 'study', 'scientists', 'discovery'],
            "health": ['health', 'medical', 'doctor', 'hospital', 'disease', 'medicine'],
            "sports": ['sports', 'game', 'team', 'player', 'score', 'championship'],
            "entertainment": ['movie', 'film', 'music', 'actor', 'celebrity', 'show'],
            "world": ['world', 'international', 'global', 'country', 'nation']
        }

        # Simple neural network for sentiment
        self.sentiment_model = self._create_sentiment_model()

        logger.info("Local classifier initialized successfully")

    def _create_sentiment_model(self):
        """Create a simple neural network for sentiment"""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 2),  # 2 classes: positive/negative
            nn.Softmax(dim=1)
        )
        return model

    def is_ready(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "local-news-classifier"

    def get_categories(self) -> list:
        return self.categories

    def encode_text(self, text: str) -> np.ndarray:
        """Get embeddings for text"""
        return self.embedding_model.encode(text)

    def classify(self, text: str) -> Dict:
        """Classify text into news categories using keyword matching"""
        processed = self.preprocessor.process(text)
        processed_lower = processed.lower()

        scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in processed_lower:
                    score += 1

            # Add some randomness for demo purposes
            scores[category] = score + np.random.random() * 0.5

        # Get best category
        best_category = max(scores, key=scores.get)
        confidence = min(scores[best_category] / 10, 0.95)  # Normalize to 0-0.95

        # Ensure minimum confidence
        confidence = max(confidence, 0.5)

        return {
            "category": best_category,
            "confidence": float(confidence),
            "scores": scores
        }

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using simple rules"""
        processed = self.preprocessor.process(text)
        processed_lower = processed.lower()

        # Simple sentiment keywords
        positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'success', 'profit']
        negative_words = ['bad', 'poor', 'negative', 'unhappy', 'failure', 'loss', 'problem']

        pos_count = sum(1 for word in positive_words if word in processed_lower)
        neg_count = sum(1 for word in negative_words if word in processed_lower)

        total = pos_count + neg_count
        if total > 0:
            pos_score = pos_count / total
            neg_score = neg_count / total
        else:
            pos_score = 0.5
            neg_score = 0.5

        if pos_score > neg_score:
            label = "positive"
            score = pos_score
        elif neg_score > pos_score:
            label = "negative"
            score = neg_score
        else:
            label = "neutral"
            score = 0.5

        # Add some randomness
        score = score * 0.8 + np.random.random() * 0.2

        return {
            "label": label,
            "score": float(score)
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts"""
        results = []
        for text in texts:
            classification = self.classify(text)
            sentiment = self.analyze_sentiment(text)
            results.append({
                "text": text,
                "classification": classification,
                "sentiment": sentiment
            })
        return results


if __name__ == "__main__":
    # Test the model
    classifier = LocalNewsClassifier()
    test_text = "The stock market reached new highs today as technology companies reported strong earnings."

    processed = classifier.preprocessor.process(test_text)
    print(f"Processed: {processed}")

    classification = classifier.classify(test_text)
    print(f"Classification: {classification}")

    sentiment = classifier.analyze_sentiment(test_text)
    print(f"Sentiment: {sentiment}")

    embedding = classifier.encode_text(test_text)
    print(f"Embedding shape: {embedding.shape}")
