# test_text_chunker.py

import pytest
from src.chunking.text_chunker import TextChunker

@pytest.fixture
def sample_data():
    sample_text = """
    This is the first paragraph of a scientific article. It introduces the main topic
    and sets the stage for the rest of the paper. Machine learning in finance is
    a growing field.

    Here is the second paragraph, detailing some background information. We discuss
    previous work and the motivation for our new approach. Our model aims to predict
    stock prices using novel deep learning architectures.

    Figure 1: This is a detailed caption for Figure 1, describing the model architecture.
    It shows the flow of data through various layers like CNN and LSTM.

    The third paragraph discusses the methodology. We employed a CNN-LSTM model for gold
    price time-series forecasting, leveraging historical data. The data acquisition
    involved specific APIs.

    Table 1.1: This table presents the evaluation metrics. It includes RMSE and MAE for
    different experimental setups.

    The final paragraph summarizes the findings and suggests future research directions,
    such as integrating external data sources like social media sentiment.
    """
    return {
        "text": sample_text,
        "article_id": "CI9TCB38",
        "page_number": 5
    }

def test_chunk_text_output_structure(sample_data):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_data["text"], sample_data["article_id"], sample_data["page_number"])
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "content" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["article_id"] == sample_data["article_id"]
        assert chunk["metadata"]["page_number"] == sample_data["page_number"]

def test_caption_detection(sample_data):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_data["text"], sample_data["article_id"], sample_data["page_number"])
    
    captions = [c for c in chunks if 'caption' in c['metadata']['type']]
    assert len(captions) >= 2  # Must include at least one figure and one table
    types = set(c['metadata']['type'] for c in captions)
    assert 'figure_caption' in types
    assert 'table_caption' in types
