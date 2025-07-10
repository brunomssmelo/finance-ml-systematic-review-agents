
import pytest
from pathlib import Path
from src.embedding.vector_store_manager import VectorStoreManager

@pytest.fixture
def dummy_chunks():
    return [
        {'content': 'LSTM-based forecasting for financial time series.', 'metadata': {'article_id': 'TEST1', 'page_number': 1, 'type': 'paragraph'}},
        {'content': 'Figure 2: CNN-LSTM model diagram.', 'metadata': {'article_id': 'TEST1', 'page_number': 2, 'type': 'figure_caption'}},
        {'content': 'Table 1: Evaluation using RMSE and MAE.', 'metadata': {'article_id': 'TEST1', 'page_number': 3, 'type': 'table_caption'}}
    ]

@pytest.fixture
def store(tmp_path):
    store = VectorStoreManager(db_path=str(tmp_path / "chroma_test"))
    store.reset_collection()
    return store

def test_add_chunks(store, dummy_chunks):
    store.add_chunks(dummy_chunks)
    # No exception means success
    assert True

def test_query_vector_store(store, dummy_chunks):
    store.add_chunks(dummy_chunks)
    query = "time series forecasting"
    results = store.query_vector_store(query_text=query, n_results=2)
    assert len(results) >= 1
    assert 'document' in results[0]
    assert 'metadata' in results[0]
    assert 'distance' in results[0]

def test_query_with_filter(store, dummy_chunks):
    store.add_chunks(dummy_chunks)
    query = "model diagram"
    filter_metadata = {'type': 'figure_caption'}
    results = store.query_vector_store(query_text=query, n_results=1, filter_metadata=filter_metadata)
    assert len(results) == 1
    assert results[0]['metadata']['type'] == 'figure_caption'

def test_reset_collection(store, dummy_chunks):
    store.add_chunks(dummy_chunks)
    store.reset_collection()
    results = store.query_vector_store(query_text="anything", n_results=1)
    assert results == []
