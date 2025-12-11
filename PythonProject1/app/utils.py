import json
import logging
import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.3f} seconds")
        return False

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def generate_article_id(text: str, prefix: str = "article") -> str:
    """
    Generate a unique article ID based on text content and timestamp

    Args:
        text: Article text
        prefix: ID prefix

    Returns:
        Unique article ID
    """
    # Create hash from text
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{prefix}_{timestamp}_{text_hash}"


def validate_text_length(text: str, min_length: int = 10, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate text length

    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"

    text_len = len(text.strip())

    if text_len < min_length:
        return False, f"Text too short. Minimum length is {min_length} characters"

    if text_len > max_length:
        return False, f"Text too long. Maximum length is {max_length} characters"

    return True, ""


def truncate_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length

    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Add "..." at the end

    Returns:
        Truncated text
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    if add_ellipsis:
        # Try to truncate at word boundary
        if truncated[-1] != ' ' and text[max_length] != ' ':
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Only if we're not losing too much
                truncated = truncated[:last_space]

        truncated = truncated.rstrip() + "..."

    return truncated


def format_confidence(confidence: float, decimal_places: int = 2) -> str:
    """
    Format confidence score as percentage

    Args:
        confidence: Confidence score (0.0 to 1.0)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if confidence < 0 or confidence > 1:
        return "N/A"

    percentage = confidence * 100
    return f"{percentage:.{decimal_places}f}%"


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize object to JSON-compatible format

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_json_serialize(item) for item in obj)
    elif isinstance(obj, set):
        return list(safe_json_serialize(item) for item in obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'dict') and callable(obj.dict):
        return safe_json_serialize(obj.dict())
    elif hasattr(obj, '__dict__'):
        return safe_json_serialize(obj.__dict__)
    else:
        return str(obj)


def save_to_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Save data to JSON file

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation

    Returns:
        Success status
    """
    try:
        serialized_data = safe_json_serialize(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=indent)

        logger.info(f"Data saved to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False


def load_from_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file

    Args:
        filepath: Input file path

    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Data loaded from {filepath}")
        return data

    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return None


def calculate_batch_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for batch processing results

    Args:
        results: List of classification results

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "avg_confidence": 0.0,
            "category_distribution": {},
            "sentiment_distribution": {}
        }

    total = len(results)
    successful = sum(1 for r in results if r.get("confidence", 0) > 0)
    failed = total - successful

    # Calculate average confidence
    confidences = [r.get("confidence", 0) for r in results if r.get("confidence", 0) > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Category distribution
    category_dist = {}
    for r in results:
        category = r.get("category", "unknown")
        category_dist[category] = category_dist.get(category, 0) + 1

    # Sentiment distribution
    sentiment_dist = {}
    for r in results:
        sentiment = r.get("sentiment", {}).get("label", "unknown") if isinstance(r.get("sentiment"),
                                                                                 dict) else "unknown"
        sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total if total > 0 else 0.0,
        "avg_confidence": avg_confidence,
        "category_distribution": category_dist,
        "sentiment_distribution": sentiment_dist
    }


def validate_api_key(api_key: Optional[str], valid_keys: List[str]) -> bool:
    """
    Validate API key

    Args:
        api_key: Provided API key
        valid_keys: List of valid API keys

    Returns:
        True if valid, False otherwise
    """
    if not valid_keys:  # No API key required
        return True

    if not api_key:
        return False

    return api_key in valid_keys


def handle_api_error(
        error: Exception,
        status_code: int = 500,
        error_type: str = "InternalServerError"
) -> HTTPException:
    """
    Handle API errors consistently

    Args:
        error: Exception object
        status_code: HTTP status code
        error_type: Error type identifier

    Returns:
        HTTPException to raise
    """
    error_message = str(error)
    logger.error(f"{error_type}: {error_message}")

    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
    )


def prepare_search_results(
        raw_results: List[Dict[str, Any]],
        max_text_length: int = 400
) -> List[Dict[str, Any]]:
    """
    Prepare search results for API response

    Args:
        raw_results: Raw search results from vector DB
        max_text_length: Maximum text length for display

    Returns:
        Formatted search results
    """
    formatted_results = []

    for result in raw_results:
        # Extract text and truncate if necessary
        text = result.get("text", "")
        if len(text) > max_text_length:
            text = truncate_text(text, max_text_length)

        # Extract metadata
        metadata = result.get("metadata", {})

        formatted_result = {
            "article_id": result.get("article_id", ""),
            "text": text,
            "category": metadata.get("category", "unknown"),
            "sentiment": metadata.get("sentiment", "unknown"),
            "similarity_score": float(result.get("similarity", 0.0)),
            "source": metadata.get("source", "Unknown"),
            "created_at": metadata.get("created_at", "")
        }

        formatted_results.append(formatted_result)

    return formatted_results


def get_sample_articles() -> List[Dict[str, str]]:
    """
    Get sample news articles for testing/demo

    Returns:
        List of sample articles
    """
    return [
        {
            "text": "The stock market reached new highs today as technology companies reported strong earnings. Analysts predict continued growth in the sector.",
            "source": "Financial Times",
            "category": "business"
        },
        {
            "text": "Scientists have discovered a new exoplanet that could potentially support life. The planet is located in the habitable zone of its star.",
            "source": "Science Journal",
            "category": "science"
        },
        {
            "text": "The government announced new climate policies aimed at reducing carbon emissions by 50% over the next decade.",
            "source": "Politics Daily",
            "category": "politics"
        },
        {
            "text": "A breakthrough in quantum computing could revolutionize how we process information, according to researchers at leading universities.",
            "source": "Tech Review",
            "category": "technology"
        },
        {
            "text": "The national team secured a dramatic victory in the championship finals, bringing home the trophy after years of effort.",
            "source": "Sports Network",
            "category": "sports"
        },
        {
            "text": "Healthcare officials warn of rising cases of seasonal flu and recommend vaccination to prevent severe illness.",
            "source": "Health Bulletin",
            "category": "health"
        },
        {
            "text": "Award-winning actor announces retirement from film industry after decades of critically acclaimed performances.",
            "source": "Entertainment Weekly",
            "category": "entertainment"
        },
        {
            "text": "International leaders gather for peace talks to address ongoing conflicts and promote diplomatic solutions.",
            "source": "World News",
            "category": "world"
        }
    ]


def create_success_response(
        data: Any,
        message: str = "Success",
        operation: str = "operation"
) -> Dict[str, Any]:
    """
    Create a standardized success response

    Args:
        data: Response data
        message: Success message
        operation: Operation name

    Returns:
        Standardized response dictionary
    """
    return {
        "status": "success",
        "message": message,
        "operation": operation,
        "data": safe_json_serialize(data),
        "timestamp": datetime.now().isoformat()
    }


def create_error_response(
        error: str,
        details: Optional[str] = None,
        operation: str = "operation"
) -> Dict[str, Any]:
    """
    Create a standardized error response

    Args:
        error: Error message
        details: Additional error details
        operation: Operation name

    Returns:
        Standardized error response
    """
    response = {
        "status": "error",
        "error": error,
        "operation": operation,
        "timestamp": datetime.now().isoformat()
    }

    if details:
        response["details"] = details

    return response


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (-1 to 1)
    """
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)

    similarity = np.dot(vec1_norm, vec2_norm)

    # Ensure the value is within valid range due to floating point errors
    similarity = max(-1.0, min(1.0, similarity))

    return float(similarity)


def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp as string

    Args:
        timestamp: Datetime object (uses current time if None)
        format_str: Format string

    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()

    return timestamp.strftime(format_str)


def validate_url(url: str) -> bool:
    """
    Validate URL format

    Args:
        url: URL to validate

    Returns:
        True if valid URL format, False otherwise
    """
    import re

    # Simple URL validation pattern
    pattern = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+'  # domain
        r'[a-zA-Z]{2,}'  # TLD
        r'(:\d+)?'  # port
        r'(/.*)?$'  # path
    )

    return bool(pattern.match(url))


# Test functions
if __name__ == "__main__":
    # Test timer
    with Timer("Test Operation") as timer:
        time.sleep(0.1)
    print(f"Elapsed time: {timer.get_elapsed():.3f}s")

    # Test article ID generation
    test_text = "This is a test article"
    article_id = generate_article_id(test_text)
    print(f"Generated article ID: {article_id}")

    # Test text validation
    is_valid, message = validate_text_length("Short", min_length=10)
    print(f"Validation result: {is_valid}, Message: {message}")

    # Test truncation
    long_text = "This is a very long text that needs to be truncated to a reasonable length for display purposes."
    truncated = truncate_text(long_text, max_length=30)
    print(f"Truncated text: {truncated}")

    # Test confidence formatting
    confidence = 0.8567
    formatted = format_confidence(confidence)
    print(f"Formatted confidence: {formatted}")

    # Test batch metrics
    test_results = [
        {"category": "business", "confidence": 0.9, "sentiment": {"label": "positive", "score": 0.95}},
        {"category": "business", "confidence": 0.8, "sentiment": {"label": "negative", "score": 0.85}},
        {"category": "technology", "confidence": 0.7, "sentiment": {"label": "positive", "score": 0.75}},
    ]
    metrics = calculate_batch_metrics(test_results)
    print(f"Batch metrics: {metrics}")