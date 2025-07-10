import pytest
from src.chunking.token_text_chunker import TokenTextChunker

@pytest.fixture
def basic_text():
    return (
        "First paragraph with simple content.\n\n"
        "Second paragraph to test splitting and overlap.\n\n"
        "Figure 1: This is a sample figure caption for testing.\n\n"
        "Table 2.1: Table caption should be extracted too.\n\n"
        "Final paragraph to end the test case."
    )

def test_chunking_token_limits(basic_text):
    chunker = TokenTextChunker(chunk_size=30, chunk_overlap=5, encoding_name="cl100k_base")
    chunks = chunker.chunk_text(basic_text, article_id="TEST01", page_number=1)
    
    # Ensure all chunks respect token limit
    for chunk in chunks:
        token_count = len(chunker.tokenizer.encode(chunk["content"]))
        assert token_count <= chunker.chunk_size

def test_caption_extraction(basic_text):
    chunker = TokenTextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_text(basic_text, article_id="TEST02", page_number=2)
    
    figure_chunks = [c for c in chunks if c["metadata"]["type"] == "figure_caption"]
    table_chunks = [c for c in chunks if c["metadata"]["type"] == "table_caption"]
    
    assert len(figure_chunks) == 1
    assert len(table_chunks) == 1
    assert "Figure 1" not in figure_chunks[0]["content"]
    assert "Table 2.1" not in table_chunks[0]["content"]

def test_metadata_inclusion(basic_text):
    chunker = TokenTextChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_text(basic_text, article_id="XYZ", page_number=42)

    for chunk in chunks:
        meta = chunk["metadata"]
        assert "article_id" in meta
        assert "page_number" in meta
        assert meta["article_id"] == "XYZ"
        assert meta["page_number"] == 42

def test_invalid_configuration():
    with pytest.raises(ValueError):
        TokenTextChunker(chunk_size=100, chunk_overlap=200)  # Invalid: overlap > size

def test_long_paragraph_splitting():
    long_text = " ".join(["token"] * 1000)  # Single paragraph with 1000 short tokens
    chunker = TokenTextChunker(chunk_size=256, chunk_overlap=32)
    chunks = chunker.chunk_text(long_text, article_id="LONG", page_number=99)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunker.tokenizer.encode(chunk["content"])) <= chunker.chunk_size
