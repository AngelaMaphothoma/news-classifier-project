import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional
import json

# API Configuration
API_BASE_URL = "http://localhost:8001"  # Change if your API runs on different port/URL

# Page Configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4a6fa5 0%, #2c3e50 100%);
        color: white;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #4a6fa5, #2c3e50);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(74, 111, 165, 0.3);
    }

    /* Text areas */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }

    /* Success message */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classified_articles' not in st.session_state:
    st.session_state.classified_articles = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"


def check_api_health() -> Dict:
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "healthy"
            return response.json()
        else:
            st.session_state.api_status = "unhealthy"
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        st.session_state.api_status = "offline"
        return {"status": "offline", "error": str(e)}


def classify_article(text: str, source: Optional[str] = None, language: str = "en") -> Optional[Dict]:
    """Send article to API for classification"""
    try:
        payload = {
            "text": text,
            "source": source,
            "language": language
        }

        response = requests.post(
            f"{API_BASE_URL}/classify",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Classification failed: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def batch_classify_articles(articles: List[Dict]) -> Optional[Dict]:
    """Send multiple articles for batch classification"""
    try:
        payload = {
            "articles": articles,
            "store_in_db": True
        }

        response = requests.post(
            f"{API_BASE_URL}/batch-classify",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Batch classification failed: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def semantic_search(query: str, top_k: int = 5) -> Optional[Dict]:
    """Perform semantic search"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "similarity_threshold": 0.7
        }

        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def get_system_stats() -> Optional[Dict]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def create_category_chart(data: List[Dict]) -> go.Figure:
    """Create category distribution chart"""
    if not data:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    df = pd.DataFrame(data)
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    fig = px.bar(
        category_counts,
        x='category',
        y='count',
        title="Article Categories Distribution",
        color='category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Category",
        yaxis_title="Count",
        showlegend=False
    )

    return fig


def create_sentiment_chart(data: List[Dict]) -> go.Figure:
    """Create sentiment distribution chart"""
    if not data:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    df = pd.DataFrame(data)

    # Extract sentiment from nested structure
    sentiments = []
    for item in data:
        if 'sentiment' in item and isinstance(item['sentiment'], dict):
            sentiments.append(item['sentiment'].get('label', 'unknown'))
        else:
            sentiments.append('unknown')

    sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    fig = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title="Sentiment Distribution",
        color='sentiment',
        color_discrete_map={
            'positive': '#4CAF50',
            'negative': '#F44336',
            'neutral': '#FFC107',
            'unknown': '#9E9E9E'
        }
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_confidence_histogram(data: List[Dict]) -> go.Figure:
    """Create confidence score histogram"""
    if not data:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    confidences = [item.get('confidence', 0) for item in data]

    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence Scores Distribution",
        labels={'x': 'Confidence Score', 'y': 'Frequency'}
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_range=[0, 1],
        showlegend=False
    )

    return fig


# Sidebar
with st.sidebar:
    st.title("📰 News Classifier")
    st.markdown("---")

    # API Status
    st.subheader("API Status")
    health_data = check_api_health()

    status_color = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
        "offline": "⚫"
    }.get(st.session_state.api_status, "⚪")

    st.markdown(f"**Status:** {status_color} {st.session_state.api_status.upper()}")

    if st.session_state.api_status == "healthy" and "components" in health_data:
        components = health_data["components"]
        st.markdown("**Components:**")
        for comp, status in components.items():
            status_icon = "✅" if status in ["ready", "connected", "operational"] else "❌"
            st.markdown(f"{status_icon} {comp}: {status}")

    st.markdown("---")

    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Single Classification", "Batch Processing", "Semantic Search", "System Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System Info
    st.subheader("System Information")

    stats = get_system_stats()
    if stats:
        if 'vector_database' in stats:
            st.metric("Articles in DB", stats['vector_database'].get('total_articles', 0))

        if 'model_info' in stats:
            st.metric("Categories", len(stats['model_info'].get('categories', [])))

    st.markdown("---")
    st.caption("Final Project - News Article Classification System")

# Main Content
if page == "Dashboard":
    st.title("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Classified", len(st.session_state.classified_articles))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("API Status", st.session_state.api_status.title())
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        search_count = len(st.session_state.search_results)
        st.metric("Search Results", search_count)
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts
    if st.session_state.classified_articles:
        col1, col2 = st.columns(2)

        with col1:
            category_chart = create_category_chart(st.session_state.classified_articles)
            st.plotly_chart(category_chart, use_container_width=True)

        with col2:
            sentiment_chart = create_sentiment_chart(st.session_state.classified_articles)
            st.plotly_chart(sentiment_chart, use_container_width=True)

        st.plotly_chart(create_confidence_histogram(st.session_state.classified_articles), use_container_width=True)

    # Recent Activity
    st.subheader("Recent Classifications")
    if st.session_state.classified_articles:
        recent_data = []
        for article in st.session_state.classified_articles[-5:]:  # Last 5 articles
            recent_data.append({
                "ID": article.get('article_id', 'N/A')[:15] + "...",
                "Category": article.get('category', 'N/A'),
                "Confidence": f"{article.get('confidence', 0) * 100:.1f}%",
                "Sentiment": article.get('sentiment', {}).get('label', 'N/A') if isinstance(article.get('sentiment'),
                                                                                            dict) else 'N/A',
                "Timestamp": article.get('timestamp', 'N/A')[:19]
            })

        st.dataframe(pd.DataFrame(recent_data), use_container_width=True)
    else:
        st.info("No classifications yet. Use the Single Classification or Batch Processing pages to start.")

elif page == "Single Classification":
    st.title("Single Article Classification")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        article_text = st.text_area(
            "Enter News Article",
            height=200,
            placeholder="Paste or type your news article here...",
            help="Enter the full text of the news article for classification"
        )

        # Additional options
        with st.expander("Additional Options"):
            source = st.text_input("Source (Optional)", placeholder="e.g., New York Times, BBC")
            language = st.selectbox("Language", ["en", "es", "fr", "de", "other"], index=0)

        # Classify button
        if st.button("Classify Article", type="primary", use_container_width=True):
            if article_text.strip():
                with st.spinner("Classifying article..."):
                    result = classify_article(article_text, source, language)

                    if result:
                        # Store in session state
                        st.session_state.classified_articles.append(result)

                        # Display results
                        st.success("Classification complete!")

                        # Results in columns
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Category", result['category'].upper())
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col_b:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            confidence_pct = result['confidence'] * 100
                            st.metric("Confidence", f"{confidence_pct:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col_c:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            sentiment = result['sentiment']['label'] if isinstance(result['sentiment'], dict) else \
                            result['sentiment']
                            st.metric("Sentiment", sentiment.upper())
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Show processed text
                        with st.expander("View Processed Text"):
                            st.write(result['processed_text'])

                        # Show similar articles if available
                        if result.get('similar_articles'):
                            st.subheader("Similar Articles Found")
                            for i, similar in enumerate(result['similar_articles'], 1):
                                st.write(f"{i}. {similar}")
            else:
                st.warning("Please enter article text to classify.")

    with col2:
        st.subheader("Quick Examples")

        examples = [
            "The stock market reached record highs as tech companies reported strong quarterly earnings.",
            "Scientists discovered a new species of deep-sea fish with bioluminescent features.",
            "Government announces new environmental policies to reduce carbon emissions by 2030.",
            "Local sports team wins championship after dramatic final match."
        ]

        for example in examples:
            if st.button(example[:60] + "...", key=example[:20]):
                st.session_state.example_text = example

        if 'example_text' in st.session_state:
            article_text = st.text_area(
                "Example Text",
                value=st.session_state.example_text,
                height=150
            )

elif page == "Batch Processing":
    st.title("Batch Article Processing")

    tab1, tab2 = st.tabs(["Upload JSON", "Manual Entry"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload JSON file with articles",
            type=['json'],
            help="Upload a JSON file containing an array of articles with 'text' and optional 'source' fields"
        )

        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)

                if isinstance(data, list):
                    st.success(f"Loaded {len(data)} articles from file")

                    # Display preview
                    with st.expander("Preview Articles"):
                        preview_df = pd.DataFrame(data[:5])  # Show first 5
                        st.dataframe(preview_df)

                    if st.button("Process All Articles", type="primary"):
                        with st.spinner(f"Processing {len(data)} articles..."):
                            result = batch_classify_articles(data)

                            if result:
                                st.success(
                                    f"Processed {result['total_processed']} articles in {result['processing_time']:.2f} seconds")

                                # Add to session state
                                for article_result in result['results']:
                                    st.session_state.classified_articles.append(article_result)

                                # Show summary
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Total Processed", result['total_processed'])

                                with col2:
                                    avg_time = result['processing_time'] / result['total_processed'] if result[
                                                                                                            'total_processed'] > 0 else 0
                                    st.metric("Avg Time/Article", f"{avg_time:.2f}s")

                                with col3:
                                    categories = set([r['category'] for r in result['results']])
                                    st.metric("Unique Categories", len(categories))

                else:
                    st.error("JSON file should contain an array of articles")

            except json.JSONDecodeError:
                st.error("Invalid JSON file")

    with tab2:
        st.write("Enter multiple articles (one per line) or paste JSON array")

        manual_input = st.text_area(
            "Enter articles (JSON format)",
            height=300,
            placeholder='[{"text": "Article 1", "source": "Source 1"}, {"text": "Article 2", "source": "Source 2"}]'
        )

        if manual_input.strip():
            try:
                data = json.loads(manual_input)

                if isinstance(data, list):
                    st.info(f"Found {len(data)} articles in input")

                    if st.button("Process Entered Articles", type="primary"):
                        with st.spinner(f"Processing {len(data)} articles..."):
                            result = batch_classify_articles(data)

                            if result:
                                st.success(f"Processed {result['total_processed']} articles")

                                # Add to session state
                                for article_result in result['results']:
                                    st.session_state.classified_articles.append(article_result)

                else:
                    st.error("Input should be a JSON array of articles")

            except json.JSONDecodeError:
                st.error("Invalid JSON format")

elif page == "Semantic Search":
    st.title("Semantic Search")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter search terms or describe what you're looking for..."
        )

    with col2:
        top_k = st.slider("Results to show", 1, 20, 5)

    if st.button("Search", type="primary"):
        if search_query.strip():
            with st.spinner("Searching..."):
                results = semantic_search(search_query, top_k)

                if results and results.get('results'):
                    st.session_state.search_results = results['results']

                    st.success(f"Found {results['total_found']} results")

                    # Display results
                    for i, result in enumerate(results['results'], 1):
                        with st.container():
                            st.markdown(f"**Result {i}** (Similarity: {result['similarity_score']:.3f})")

                            col_a, col_b = st.columns([3, 1])

                            with col_a:
                                st.write(result['text'])

                            with col_b:
                                st.metric("Category", result['category'].upper())
                                st.metric("Sentiment", result['sentiment'].upper())

                            if result.get('source') and result['source'] != "Unknown":
                                st.caption(f"Source: {result['source']}")

                            st.divider()
                else:
                    st.info("No results found. Try different search terms.")
        else:
            st.warning("Please enter a search query.")

    # Show recent search results if available
    if st.session_state.search_results:
        st.subheader("Recent Search Results")

        results_df = pd.DataFrame(st.session_state.search_results)

        # Select columns to display
        display_cols = ['text', 'category', 'sentiment', 'similarity_score']
        available_cols = [col for col in display_cols if col in results_df.columns]

        if available_cols:
            st.dataframe(results_df[available_cols], use_container_width=True)

elif page == "System Analytics":
    st.title("System Analytics")

    stats = get_system_stats()

    if stats:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vector Database")

            if 'vector_database' in stats:
                db_stats = stats['vector_database']

                st.metric("Total Articles", db_stats.get('total_articles', 0))
                st.metric("Embedding Dimension", db_stats.get('embedding_dimension', 0))

                if db_stats.get('categories'):
                    st.write("**Categories in DB:**")
                    for category in db_stats['categories']:
                        st.write(f"• {category}")

        with col2:
            st.subheader("Model Information")

            if 'model_info' in stats:
                model_info = stats['model_info']

                st.metric("Model Name", model_info.get('name', 'N/A'))
                st.metric("Max Input Length", model_info.get('max_input_length', 0))

                if model_info.get('categories'):
                    st.write("**Supported Categories:**")
                    for category in model_info['categories']:
                        st.write(f"• {category}")

    # Classification Analytics
    st.subheader("Classification Analytics")

    if st.session_state.classified_articles:
        df = pd.DataFrame(st.session_state.classified_articles)

        # Convert sentiment to simple label
        sentiments = []
        for item in st.session_state.classified_articles:
            if 'sentiment' in item and isinstance(item['sentiment'], dict):
                sentiments.append(item['sentiment'].get('label', 'unknown'))
            else:
                sentiments.append('unknown')

        df['sentiment_label'] = sentiments

        col1, col2 = st.columns(2)

        with col1:
            # Category distribution
            category_dist = df['category'].value_counts()
            st.bar_chart(category_dist)

        with col2:
            # Sentiment distribution
            sentiment_dist = df['sentiment_label'].value_counts()
            st.bar_chart(sentiment_dist)

        # Confidence statistics
        st.metric("Average Confidence", f"{df['confidence'].mean() * 100:.1f}%")

        # Recent activity timeline
        st.subheader("Recent Activity")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)

            recent_activity = df.head(10)[['timestamp', 'category', 'confidence', 'sentiment_label']]
            recent_activity['confidence'] = recent_activity['confidence'].apply(lambda x: f"{x * 100:.1f}%")
            recent_activity['timestamp'] = recent_activity['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(recent_activity, use_container_width=True)

    else:
        st.info("No classification data available yet.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption("News Article Classification System • Final Project • Powered by FastAPI & Streamlit")