import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import logging
import uuid
import json

from app.schemas import Category, Sentiment, ArticleMetadata, ArticleDocument
from app.models import NewsClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database client
_chroma_client = None
_collection = None
_embedding_function = None


def get_vector_db():
    """Get or initialize the vector database client"""
    global _chroma_client, _collection, _embedding_function

    if _chroma_client is None:
        init_vector_db()

    return {
        "client": _chroma_client,
        "collection": _collection,
        "embedding_function": _embedding_function
    }


async def init_vector_db(persist_directory: str = "./chroma_db"):
    """Initialize the vector database"""
    global _chroma_client, _collection, _embedding_function

    try:
        logger.info(f"Initializing ChromaDB at {persist_directory}")

        # Create embedding function
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Create Chroma client
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        _collection = _chroma_client.get_or_create_collection(
            name="news_articles",
            embedding_function=_embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("ChromaDB initialized successfully")
        logger.info(f"Collection count: {_collection.count()}")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


class VectorDatabase:
    """Wrapper class for ChromaDB operations"""

    def __init__(self):
        self.db_info = get_vector_db()
        self.collection = self.db_info["collection"]
        self.classifier = NewsClassifier()

    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.collection is not None

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()

            # Get unique categories from metadata
            all_items = self.collection.get(include=["metadatas"])
            categories = set()

            if all_items and all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    if metadata and "category" in metadata:
                        categories.add(metadata["category"])

            return {
                "total_articles": count,
                "categories": list(categories),
                "embedding_dimension": 384  # SentenceTransformer dimension
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"total_articles": 0, "categories": [], "embedding_dimension": 384}

    def add_article(
            self,
            article_id: str,
            text: str,
            category: Category,
            sentiment: Sentiment,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an article to the vector database"""
        try:
            # Prepare metadata
            article_metadata = {
                "article_id": article_id,
                "category": category.value if isinstance(category, Category) else category,
                "sentiment": sentiment.value if isinstance(sentiment, Sentiment) else sentiment,
                "created_at": datetime.now().isoformat(),
                **metadata
            }

            # Clean metadata - ensure all values are strings or numbers
            clean_metadata = {}
            for key, value in article_metadata.items():
                if value is None:
                    clean_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)

            # Add to collection
            self.collection.add(
                documents=[text],
                metadatas=[clean_metadata],
                ids=[article_id]
            )

            logger.info(f"Article added to database: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add article {article_id}: {e}")
            return False

    def add_batch_articles(self, articles: List[ArticleDocument]) -> int:
        """Add multiple articles in batch"""
        try:
            documents = []
            metadatas = []
            ids = []

            for article in articles:
                # Prepare metadata
                metadata = {
                    "article_id": article.article_id,
                    "category": article.category.value if isinstance(article.category, Category) else article.category,
                    "sentiment": article.sentiment.value if isinstance(article.sentiment,
                                                                       Sentiment) else article.sentiment,
                    "created_at": datetime.now().isoformat(),
                    **article.metadata.dict(exclude_none=True)
                }

                # Clean metadata
                clean_metadata = {}
                for key, value in metadata.items():
                    if value is None:
                        clean_metadata[key] = ""
                    elif isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)

                documents.append(article.text)
                metadatas.append(clean_metadata)
                ids.append(article.article_id)

            # Add batch
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(articles)} articles to database")
            return len(articles)

        except Exception as e:
            logger.error(f"Failed to add batch articles: {e}")
            return 0

    def semantic_search(
            self,
            query: str,
            top_k: int = 5,
            min_similarity: float = 0.7,
            exclude_id: Optional[str] = None,
            filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on articles"""
        try:
            # Prepare where filter
            where_filter = {}
            if filter_conditions:
                where_filter = filter_conditions

            # If excluding an article, add to filter
            if exclude_id:
                where_filter["article_id"] = {"$ne": exclude_id}

            # Perform query
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []

            if results and results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity score (cosine distance to similarity)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Assuming cosine distance

                    if similarity >= min_similarity:
                        formatted_results.append({
                            "article_id": results["metadatas"][0][i].get("article_id", ""),
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": similarity,
                            "distance": distance
                        })

            logger.info(f"Semantic search found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific article by ID"""
        try:
            results = self.collection.get(
                ids=[article_id],
                include=["documents", "metadatas"]
            )

            if results and results["documents"] and len(results["documents"]) > 0:
                return {
                    "article_id": article_id,
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get article {article_id}: {e}")
            return None

    def get_articles_by_category(self, category: Category, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles by category"""
        try:
            results = self.collection.get(
                where={"category": category.value if isinstance(category, Category) else category},
                limit=limit,
                include=["documents", "metadatas"]
            )

            formatted_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"])):
                    formatted_results.append({
                        "article_id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get articles by category {category}: {e}")
            return []

    def get_articles_by_sentiment(self, sentiment: Sentiment, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles by sentiment"""
        try:
            results = self.collection.get(
                where={"sentiment": sentiment.value if isinstance(sentiment, Sentiment) else sentiment},
                limit=limit,
                include=["documents", "metadatas"]
            )

            formatted_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"])):
                    formatted_results.append({
                        "article_id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get articles by sentiment {sentiment}: {e}")
            return []

    def update_article_metadata(self, article_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update article metadata"""
        try:
            # Get current article
            article = self.get_article(article_id)
            if not article:
                return False

            # Merge metadata
            current_metadata = article["metadata"]
            updated_metadata = {**current_metadata, **metadata_updates, "updated_at": datetime.now().isoformat()}

            # Update in collection
            self.collection.update(
                ids=[article_id],
                metadatas=[updated_metadata]
            )

            logger.info(f"Updated metadata for article: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update article {article_id}: {e}")
            return False

    def delete_article(self, article_id: str) -> bool:
        """Delete an article from the database"""
        try:
            self.collection.delete(ids=[article_id])
            logger.info(f"Deleted article: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all articles from the database"""
        try:
            self.collection.delete(where={})
            logger.info("Cleared all articles from database")
            return True

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def search_by_keywords(
            self,
            keywords: List[str],
            operator: str = "AND",
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search articles by keywords (traditional search)"""
        try:
            # Get all documents
            all_items = self.collection.get(
                include=["documents", "metadatas"],
                limit=1000  # Limit for performance
            )

            if not all_items or not all_items["documents"]:
                return []

            results = []

            for i, text in enumerate(all_items["documents"]):
                text_lower = text.lower()
                matches = 0

                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matches += 1

                # Check operator condition
                if operator == "AND" and matches == len(keywords):
                    results.append({
                        "article_id": all_items["ids"][i],
                        "text": text,
                        "metadata": all_items["metadatas"][i] if all_items["metadatas"] else {},
                        "keyword_matches": matches
                    })
                elif operator == "OR" and matches > 0:
                    results.append({
                        "article_id": all_items["ids"][i],
                        "text": text,
                        "metadata": all_items["metadatas"][i] if all_items["metadatas"] else {},
                        "keyword_matches": matches
                    })

            # Sort by number of matches and limit
            results.sort(key=lambda x: x["keyword_matches"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []


# Test function
if __name__ == "__main__":
    import asyncio


    async def test_database():
        """Test the database functionality"""
        await init_vector_db()
        db = VectorDatabase()

        print(f"Database connected: {db.is_connected()}")
        print(f"Statistics: {db.get_statistics()}")

        # Test adding an article
        test_id = f"test_article_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        success = db.add_article(
            article_id=test_id,
            text="The stock market reached new highs today as technology companies reported strong earnings.",
            category=Category.BUSINESS,
            sentiment=Sentiment.POSITIVE,
            metadata={"source": "Financial Times", "language": "en"}
        )

        print(f"Article added: {success}")

        # Test semantic search
        search_results = db.semantic_search("stock market performance", top_k=3)
        print(f"Search results: {len(search_results)} found")

        for result in search_results:
            print(f"  - {result['article_id']}: {result['text'][:50]}... (similarity: {result['similarity']:.3f})")

        # Test get article
        article = db.get_article(test_id)
        print(f"Retrieved article: {article is not None}")

        # Clean up
        db.delete_article(test_id)
        print(f"Test article deleted")


    asyncio.run(test_database())
