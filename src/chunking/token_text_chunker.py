# %%

import re
import tiktoken
from typing import List, Tuple, Dict, Optional

# %%
class TokenTextChunker:
    """
    A class that splits input text into token-based chunks, optionally extracting
    figure/table captions separately to preserve semantic coherence.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, encoding_name: str = "cl100k_base"):
        """
        Initializes the TokenTextChunker.

        Args:
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
            encoding_name (str): Name of the tokenizer encoding (e.g., "cl100k_base" for OpenAI models).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, article_id: str, page_number: Optional[int] = None) -> List[Dict]:
        """
        Main entry point. Extracts captions and chunks the remaining body text.

        Args:
            text (str): Raw text input.
            article_id (str): Unique identifier for the article.
            page_number (int, optional): Page number of the source text.

        Returns:
            List[Dict]: A list of chunk dictionaries with content and metadata.
        """
        caption_chunks, body_text = self._extract_captions(text, article_id, page_number)
        body_chunks = self._chunk_body_text(body_text, article_id, page_number)
        return caption_chunks + body_chunks

    def _extract_captions(self, text: str, article_id: str, page_number: Optional[int] = None) -> Tuple[List[Dict], str]:
        """
        Extracts figure/table captions from the text.

        Returns:
            Tuple:
                - A list of caption chunks (as dicts)
                - The cleaned text with captions removed
        """
        caption_pattern = r"(Figure|Table)\s+\d+(?:\.\d+)?[:\.\s]+(.*?)(?=\n{2,}|\n[A-Z]|\Z)"
        matches = list(re.finditer(caption_pattern, text, re.DOTALL))

        caption_chunks = []
        cleaned_text = text

        for match in matches:
            caption_type = match.group(1)
            caption_content = match.group(2).strip()
            full_caption = match.group(0)

            if caption_content:
                source_element = full_caption.split(':')[0].strip() if ':' in full_caption else full_caption.strip()
                caption_chunks.append({
                    "content": caption_content,
                    "metadata": {
                        "article_id": article_id,
                        "page_number": page_number,
                        "type": f"{caption_type.lower()}_caption",
                        "source_element": source_element
                    }
                })

                # Remove only the first occurrence to prevent over-deletion
                cleaned_text = cleaned_text.replace(full_caption, "", 1)

        return caption_chunks, cleaned_text

    def _chunk_body_text(self, text: str, article_id: str, page_number: Optional[int] = None) -> List[Dict]:
        """
        Splits the main body text into token-based chunks.

        Args:
            text (str): The body text to be chunked.
            article_id (str): Article identifier.
            page_number (int, optional): Page number.

        Returns:
            List[Dict]: List of token-based chunks with metadata.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        token_buffer = []

        for paragraph in paragraphs:
            para_token_ids = self.tokenizer.encode(paragraph)
            remaining_tokens = para_token_ids

            while remaining_tokens:
                space_left = self.chunk_size - len(token_buffer)

                if space_left <= 0:
                    if token_buffer:
                        chunks.append({
                            "content": self.tokenizer.decode(token_buffer),
                            "metadata": {
                                "article_id": article_id,
                                "page_number": page_number,
                                "type": "text_chunk"
                            }
                        })
                        token_buffer = token_buffer[-self.chunk_overlap:] if self.chunk_overlap > 0 else []

                tokens_to_add = min(len(remaining_tokens), self.chunk_size - len(token_buffer))
                token_buffer.extend(remaining_tokens[:tokens_to_add])
                remaining_tokens = remaining_tokens[tokens_to_add:]

                if len(token_buffer) >= self.chunk_size:
                    chunks.append({
                        "content": self.tokenizer.decode(token_buffer),
                        "metadata": {
                            "article_id": article_id,
                            "page_number": page_number,
                            "type": "text_chunk"
                        }
                    })
                    token_buffer = token_buffer[-self.chunk_overlap:] if self.chunk_overlap > 0 else []

        if token_buffer:
            chunks.append({
                "content": self.tokenizer.decode(token_buffer),
                "metadata": {
                    "article_id": article_id,
                    "page_number": page_number,
                    "type": "text_chunk"
                }
            })

        return chunks

# %%
# __name__ = "__main__"
# # Example usage
if __name__ == "__main__":
    chunker = TokenTextChunker(chunk_size=500, chunk_overlap=50, encoding_name="cl100k_base")

    sample_text = """
    This is the first paragraph of a scientific article. It introduces the main topic
    and sets the stage for the rest of the work. Machine learning in finance is a growing field.

    Here's the second paragraph, detailing some background information. We discuss
    previous work and the motivation for our novel approach. Our model aims to predict
    stock prices using deep learning architectures.

    Figure 1: This is a detailed caption for Figure 1, describing the model architecture.
    It shows data flow through layers like CNN and LSTM.

    The third paragraph discusses methodology. We used a CNN-LSTM model for forecasting
    gold price time series based on historical data. Data acquisition involved specific APIs.

    Table 1.1: This table presents evaluation metrics including RMSE and MAE
    for various experimental configurations.

    The final paragraph summarizes the findings and suggests directions for future work,
    such as integrating external data sources like social media sentiment.

    This is a very, very long paragraph added to test how the chunker behaves when a single
    paragraph exceeds the defined chunk_size. In real-world scenarios, scientific articles
    can contain dense sections or detailed experimental setups that, when tokenized,
    easily exceed the token limit for a single chunk. The chunker must be capable of
    breaking large paragraphs into multiple overlapping sub-chunks to preserve context
    across boundaries. This is crucial for large language models (LLMs), which have
    limited context windows and token-based pricing. Poor chunking can lead to
    truncation or inefficient token usage. The overlap helps retain important
    contextual signals between chunk boundaries, especially for downstream RAG agents.
    """

    article_id = "CI9TCB38"
    page_number = 5

    chunks = chunker.chunk_text(sample_text, article_id, page_number)

    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Content (first 250 chars): {chunk['content'][:250]}...")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Token count: {len(chunker.tokenizer.encode(chunk['content']))}")
