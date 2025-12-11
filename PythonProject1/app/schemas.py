from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Category(str, Enum):
    """News categories"""
    POLITICS = "politics"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    HEALTH = "health"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    WORLD = "world"
    GENERAL = "general"


class Sentiment(str, Enum):
    """Sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ArticleRequest(BaseModel):
    """Request model for single article classification"""
    text: str = Field(..., min_length=10, max_length=10000, description="News article text")
    source: Optional[str] = Field(None, description="Source publication")
    language: Optional[str] = Field("en", description="Language code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The stock market reached new highs today as technology companies reported strong earnings.",
                "source": "Financial Times",
                "language": "en"
            }
        }
    )


class ClassificationResult(BaseModel):
    """Single classification result"""
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    all_probabilities: Optional[Dict[str, float]] = None


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: Sentiment
    score: float = Field(..., ge=0.0, le=1.0, description="Sentiment score")


class ClassificationResponse(BaseModel):
    """Response model for article classification"""
    article_id: str
    original_text: str
    processed_text: str
    category: Category
    confidence: float
    sentiment: SentimentResult
    timestamp: str
    similar_articles: Optional[List[str]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "article_id": "article_20241215_abc123",
                "original_text": "The stock market reached new highs...",
                "processed_text": "stock market reached new highs technology companies reported strong earnings",
                "category": "business",
                "confidence": 0.85,
                "sentiment": {
                    "label": "positive",
                    "score": 0.92
                },
                "timestamp": "2024-12-15T10:30:00",
                "similar_articles": [
                    "Market analysts predict continued growth in tech sector",
                    "Quarterly earnings exceed expectations for major companies"
                ]
            }
        }
    )


class BatchRequest(BaseModel):
    """Request model for batch classification"""
    articles: List[ArticleRequest] = Field(..., min_items=1, max_items=100, description="List of articles")
    store_in_db: Optional[bool] = Field(True, description="Store results in database")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "articles": [
                    {
                        "text": "The stock market reached new highs today.",
                        "source": "Financial Times",
                        "language": "en"
                    },
                    {
                        "text": "New scientific discovery could revolutionize medicine.",
                        "source": "Science Journal",
                        "language": "en"
                    }
                ],
                "store_in_db": True
            }
        }
    )


class BatchResponse(BaseModel):
    """Response model for batch classification"""
    results: List[ClassificationResponse]
    total_processed: int
    processing_time: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "article_id": "article_20241215_abc123",
                        "original_text": "The stock market reached new highs...",
                        "processed_text": "stock market reached new highs",
                        "category": "business",
                        "confidence": 0.85,
                        "sentiment": {"label": "positive", "score": 0.92},
                        "timestamp": "2024-12-15T10:30:00"
                    }
                ],
                "total_processed": 1,
                "processing_time": 0.45
            }
        }
    )


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., min_length=3, description="Search query")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "stock market performance",
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        }
    )


class SearchResult(BaseModel):
    """Single search result"""
    article_id: str
    text: str
    category: Category
    sentiment: Sentiment
    similarity_score: float
    source: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "article_id": "article_20241215_abc123",
                "text": "The stock market reached new highs...",
                "category": "business",
                "sentiment": "positive",
                "similarity_score": 0.85,
                "source": "Financial Times"
            }
        }
    )


class SearchResponse(BaseModel):
    """Response model for semantic search"""
    query: str
    results: List[SearchResult]
    total_found: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "stock market performance",
                "results": [
                    {
                        "article_id": "article_20241215_abc123",
                        "text": "The stock market reached new highs...",
                        "category": "business",
                        "sentiment": "positive",
                        "similarity_score": 0.85,
                        "source": "Financial Times"
                    }
                ],
                "total_found": 1
            }
        }
    )


class ComponentStatus(BaseModel):
    """Individual component status"""
    ml_model: str
    vector_database: str
    preprocessing: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: ComponentStatus
    timestamp: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "components": {
                    "ml_model": "ready",
                    "vector_database": "connected",
                    "preprocessing": "operational"
                },
                "timestamp": "2024-12-15T10:30:00"
            }
        }
    )


class VectorDBStats(BaseModel):
    """Vector database statistics"""
    total_articles: int
    categories: List[str]
    embedding_dimension: int


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    categories: List[Category]
    max_input_length: int


class StatsResponse(BaseModel):
    """Statistics response"""
    vector_database: VectorDBStats
    model_info: ModelInfo

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vector_database": {
                    "total_articles": 150,
                    "categories": ["politics", "business", "technology"],
                    "embedding_dimension": 384
                },
                "model_info": {
                    "name": "distilbert-base-uncased",
                    "type": "transformer",
                    "categories": ["politics", "business", "technology", "science", "health", "sports", "entertainment",
                                   "world"],
                    "max_input_length": 512
                }
            }
        }
    )


class ArticleMetadata(BaseModel):
    """Article metadata for database storage"""
    source: Optional[str] = None
    language: Optional[str] = "en"
    original_length: Optional[int] = None
    processed_length: Optional[int] = None
    batch_processed: Optional[bool] = False
    category: Optional[Category] = None
    sentiment: Optional[Sentiment] = None


class ArticleDocument(BaseModel):
    """Complete article document for database"""
    article_id: str
    text: str
    category: Category
    sentiment: Sentiment
    metadata: ArticleMetadata
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    error_type: Optional[str] = None
    timestamp: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Classification failed: Text too short",
                "error_type": "ValidationError",
                "timestamp": "2024-12-15T10:30:00"
            }
        }
    )