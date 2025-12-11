from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from app.database import get_vector_db, init_vector_db
from app.models import NewsClassifier, TextPreprocessor
from app.schemas import (
    ArticleRequest, ClassificationResponse,
    BatchRequest, BatchResponse,
    SearchRequest, SearchResponse,
    HealthResponse, StatsResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
classifier = None
preprocessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier, preprocessor
    logger.info("Starting up application...")

    try:
        # Initialize components
        preprocessor = TextPreprocessor()
        classifier = NewsClassifier()
        await init_vector_db()

        logger.info("✅ Application started successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    finally:
        # Shutdown
        logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="News Article Classification API",
    description="API for classifying news articles and performing sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "News Article Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    try:
        model_status = classifier.is_ready() if classifier else False
        db_status = get_vector_db().is_connected()

        return HealthResponse(
            status="healthy" if model_status and db_status else "degraded",
            components={
                "ml_model": "ready" if model_status else "unavailable",
                "vector_database": "connected" if db_status else "disconnected",
                "preprocessing": "operational" if preprocessor else "unavailable"
            },
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system statistics"""
    try:
        vector_db = get_vector_db()
        db_stats = vector_db.get_statistics()

        return StatsResponse(
            vector_database={
                "total_articles": db_stats.get("total_articles", 0),
                "categories": db_stats.get("categories", []),
                "embedding_dimension": 384  # sentence-transformers dimension
            },
            model_info={
                "name": classifier.get_model_name(),
                "type": "transformer",
                "categories": classifier.get_categories(),
                "max_input_length": 512
            }
        )
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.post("/classify", response_model=ClassificationResponse)
async def classify_article(request: ArticleRequest):
    """
    Classify a single news article

    - **text**: News article text (required)
    - **source**: Source publication (optional)
    - **language**: Language code (default: "en")
    """
    if not classifier or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Preprocess text
        processed_text = preprocessor.process(request.text)

        # Get classification
        classification = classifier.classify(processed_text)

        # Get sentiment analysis
        sentiment = classifier.analyze_sentiment(processed_text)

        # Generate article ID
        import hashlib
        text_hash = hashlib.md5(processed_text.encode()).hexdigest()[:8]
        article_id = f"article_{datetime.now().strftime('%Y%m%d')}_{text_hash}"

        # Store in vector database
        vector_db = get_vector_db()
        vector_db.add_article(
            article_id=article_id,
            text=processed_text,
            category=classification["category"],
            sentiment=sentiment["label"],
            metadata={
                "source": request.source,
                "language": request.language,
                "original_length": len(request.text),
                "processed_length": len(processed_text)
            }
        )

        # Find similar articles
        similar = vector_db.semantic_search(
            query=processed_text,
            top_k=3,
            exclude_id=article_id
        )
        similar_articles = [result["text"] for result in similar]

        return ClassificationResponse(
            article_id=article_id,
            original_text=request.text[:500] + "..." if len(request.text) > 500 else request.text,
            processed_text=processed_text[:500] + "..." if len(processed_text) > 500 else processed_text,
            category=classification["category"],
            confidence=classification["confidence"],
            sentiment=sentiment["label"],
            sentiment_score=sentiment["score"],
            timestamp=datetime.now().isoformat(),
            similar_articles=similar_articles if similar_articles else None
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/batch-classify", response_model=BatchResponse)
async def batch_classify_articles(request: BatchRequest):
    """
    Classify multiple news articles in batch
    """
    if not classifier or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not ready")

    import time
    start_time = time.time()
    results = []
    successful = 0

    for i, article in enumerate(request.articles):
        try:
            processed_text = preprocessor.process(article.text)
            classification = classifier.classify(processed_text)
            sentiment = classifier.analyze_sentiment(processed_text)

            import hashlib
            text_hash = hashlib.md5(processed_text.encode()).hexdigest()[:8]
            article_id = f"batch_{datetime.now().strftime('%Y%m%d')}_{i:04d}_{text_hash}"

            vector_db = get_vector_db()
            if request.store_in_db:
                vector_db.add_article(
                    article_id=article_id,
                    text=processed_text,
                    category=classification["category"],
                    sentiment=sentiment["label"],
                    metadata={
                        "source": article.source,
                        "language": article.language,
                        "batch_processed": True
                    }
                )

            response = ClassificationResponse(
                article_id=article_id,
                original_text=article.text[:300] + "..." if len(article.text) > 300 else article.text,
                processed_text=processed_text[:300] + "..." if len(processed_text) > 300 else processed_text,
                category=classification["category"],
                confidence=classification["confidence"],
                sentiment=sentiment["label"],
                sentiment_score=sentiment["score"],
                timestamp=datetime.now().isoformat()
            )

            results.append(response)
            successful += 1

        except Exception as e:
            logger.warning(f"Failed to process article {i}: {e}")
            continue

    processing_time = time.time() - start_time

    return BatchResponse(
        results=results,
        total_processed=successful,
        processing_time=processing_time
    )


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on stored articles
    """
    try:
        processed_query = preprocessor.process(request.query)
        vector_db = get_vector_db()

        results = vector_db.semantic_search(
            query=processed_query,
            top_k=request.top_k,
            min_similarity=request.similarity_threshold
        )

        formatted_results = []
        for result in results:
            formatted_results.append({
                "article_id": result["article_id"],
                "text": result["text"][:400] + "..." if len(result["text"]) > 400 else result["text"],
                "category": result["metadata"]["category"],
                "sentiment": result["metadata"]["sentiment"],
                "similarity_score": float(result["similarity"]),
                "source": result["metadata"].get("source", "Unknown")
            })

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_found=len(formatted_results)
        )

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )