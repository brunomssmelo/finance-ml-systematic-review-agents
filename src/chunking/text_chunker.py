import re

class TextChunker:
    """
    A class to chunk text into smaller, semantically coherent pieces.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initializes the TextChunker with specified chunk size and overlap.

        Args:
            chunk_size (int): The target maximum size of each chunk (in words or tokens).
                              Recommended: 200 to 500 tokens.
            chunk_overlap (int): The number of words/tokens to overlap between consecutive chunks.
                                 Recommended: 10-20% of chunk size.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, article_id: str, page_number: int = None) -> list[dict]:
        """
        Splits text into chunks, prioritizing natural boundaries and preserving metadata.
        Also specifically handles figure/table captions.

        Args:
            text (str): The full text content of a page or section.
            article_id (str): The unique ID of the article (from PAPER.ID) [2, Previous Turn].
            page_number (int, optional): The page number from which the text was extracted.

        Returns:
            list[dict]: A list of dictionaries, where each dict is a chunk with its content and metadata.
                        Metadata includes article_id, page_number, and potentially section/subsection.
                        Example: [{'content': '...', 'metadata': {'article_id': 'CI9TCB38', 'page_number': 1}}]
        """
        chunks = []
        
        # 1. Extract and process captions first
        # Regex to find Figure/Table captions (simple heuristic, can be improved)
        caption_pattern = r"(Figure|Table)\s+\d+(\.\d+)?[:\.\s]+(.*?)(?=\n\n|\n[A-Z]|\Z)"
        captions = re.findall(caption_pattern, text, re.DOTALL)
        
        processed_text = text # Text after removing extracted captions
        
        for match in captions:
            caption_type = match
            caption_number = match[33] # This might be empty if no sub-number
            caption_content = match[34].strip()
            
            if caption_content:
                chunks.append({
                    'content': caption_content,
                    'metadata': {
                        'article_id': article_id,
                        'page_number': page_number,
                        'type': f"{caption_type.lower()}_caption",
                        'source_element': f"{caption_type} {caption_number.strip() if caption_number else ''}".strip()
                    }
                })
                # Remove the caption from the main text to avoid re-chunking it
                processed_text = processed_text.replace(match + match[33] + match[34], "", 1)
        
        # 2. Chunk the remaining text by paragraphs
        paragraphs = [p.strip() for p in processed_text.split('\n\n') if p.strip()]

        current_chunk = []
        current_chunk_len = 0

        for para in paragraphs:
            para_words = para.split()
            para_len = len(para_words)

            if current_chunk_len + para_len <= self.chunk_size:
                current_chunk.extend(para_words)
                current_chunk_len += para_len
            else:
                if current_chunk: # Save the current chunk if it's not empty
                    chunks.append({
                        'content': " ".join(current_chunk),
                        'metadata': {
                            'article_id': article_id,
                            'page_number': page_number,
                            'type': 'text_chunk'
                            # 'section_title': section_title # Future: if derivable
                        }
                    })
                
                # Start new chunk with overlap
                start_index = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[start_index:] # Apply overlap
                current_chunk.extend(para_words)
                current_chunk_len = len(current_chunk)

        # Add the last chunk if any remaining
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