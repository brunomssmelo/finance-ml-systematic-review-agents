# %%
import re

class TextChunker:
    """
    A class to split text into smaller, semantically coherent chunks,
    optionally extracting figure/table captions separately.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initializes the TextChunker with specified chunk size and overlap.

        Args:
            chunk_size (int): The target maximum size of each chunk (in words).
            chunk_overlap (int): The number of words to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, article_id: str, page_number: int = None) -> list[dict]:
        """
        Splits text into content chunks and captions (figures/tables), preserving metadata.

        Args:
            text (str): The full text content of a page or section.
            article_id (str): Unique ID of the article.
            page_number (int, optional): The page number where the text came from.

        Returns:
            list[dict]: List of chunks with 'content' and 'metadata'.
        """
        # 1. Extract captions and remove them from text
        caption_chunks, text_without_captions = self._extract_captions(text, article_id, page_number)

        # 2. Chunk remaining text
        text_chunks = self._chunk_paragraphs(text_without_captions, article_id, page_number)

        return caption_chunks + text_chunks

    def _extract_captions(self, text: str, article_id: str, page_number: int = None) -> tuple[list[dict], str]:
        """
        Finds and extracts figure/table captions.

        Returns:
            - A list of caption chunks.
            - The text with captions removed.
        """
        caption_pattern = r"(Figure|Table)\s+\d+(?:\.\d+)?[:\.\s]+(.*?)(?=\n\n|\n[A-Z]|\Z)"
        matches = re.finditer(caption_pattern, text, re.DOTALL)

        caption_chunks = []
        cleaned_text = text

        for match in matches:
            caption_type = match.group(1)
            caption_content = match.group(2).strip()
            caption_full = match.group(0)

            if caption_content:
                source_element = caption_full.split(':')[0].strip()

                caption_chunks.append({
                    'content': caption_content,
                    'metadata': {
                        'article_id': article_id,
                        'page_number': page_number,
                        'type': f"{caption_type.lower()}_caption",
                        'source_element': source_element
                    }
                })

                cleaned_text = cleaned_text.replace(caption_full, "", 1)

        return caption_chunks, cleaned_text

    def _chunk_paragraphs(self, text: str, article_id: str, page_number: int = None) -> list[dict]:
        """
        Chunks the given text into overlapping segments by paragraphs.

        Returns:
            A list of text chunks with metadata.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []

        current_chunk = []
        current_chunk_len = 0

        for para in paragraphs:
            words = para.split()
            word_count = len(words)

            if current_chunk_len + word_count <= self.chunk_size:
                current_chunk.extend(words)
                current_chunk_len += word_count
            else:
                if current_chunk:
                    chunks.append({
                        'content': " ".join(current_chunk),
                        'metadata': {
                            'article_id': article_id,
                            'page_number': page_number,
                            'type': 'text_chunk'
                        }
                    })

                # Start new chunk with overlap
                start_index = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[start_index:] + words
                current_chunk_len = len(current_chunk)

        if current_chunk:
            chunks.append({
                'content': " ".join(current_chunk),
                'metadata': {
                    'article_id': article_id,
                    'page_number': page_number,
                    'type': 'text_chunk'
                }
            })

        return chunks

# %%
# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    
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
    
    article_id_example = "CI9TCB38" # Example ID from PAPER table [34]
    page_number_example = 5

    chunks_output = chunker.chunk_text(sample_text, article_id_example, page_number_example)

    print(f"Generated {len(chunks_output)} chunks:")
    for i, chunk in enumerate(chunks_output):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content (first 150 chars): {chunk['content'][:150]}...")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content Length (words): {len(chunk['content'].split())}")
# %%
