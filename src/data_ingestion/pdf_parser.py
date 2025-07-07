import pypdf
import os

class PdfParser:
    """
    A class to handle PDF parsing and text extraction.
    """
    def __init__(self):
        # No specific initialization needed for pypdf, but could include logging setup
        pass

    def extract_text_from_pdf(self, pdf_path: str) -> list[dict]:
        """
        Extracts text from each page of a PDF and returns it with basic metadata.

        Args:
            pdf_path (str): The full path to the PDF file.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents
                        a page and contains 'page_content' (text) and 'page_number'.
                        Example: [{'page_content': '...', 'page_number': 1}, ...]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        extracted_data = []
        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text: # Only add if text was successfully extracted
                    extracted_data.append({
                        'page_content': page_text,
                        'page_number': i + 1, # Page numbers are 1-based
                        'file_path': pdf_path # Include file path as metadata
                    })
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            # Depending on error handling strategy, could return an empty list or re-raise
            return []
        return extracted_data

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    parser = PdfParser()
    # Create a dummy PDF file for testing (or use a real one)
    # Note: You would typically have real PDFs in data/raw_pdfs/
    dummy_pdf_path = "data/raw_pdfs/example_article.pdf"

    # For demonstration, let's create a placeholder for a PDF.
    # In a real scenario, this PDF would already exist.
    if not os.path.exists("data/raw_pdfs"):
        os.makedirs("data/raw_pdfs")
    # You would need an actual PDF file here to test.
    # For now, let's assume `example_article.pdf` exists.

    # Example of how you would call it
    # page_data = parser.extract_text_from_pdf(dummy_pdf_path)
    # if page_data:
    #     print(f"Extracted {len(page_data)} pages from {dummy_pdf_path}")
    #     print(f"First page content snippet: {page_data['page_content'][:200]}...")
    #     print(f"Metadata for first page: {page_data['page_number']}, {page_data['file_path']}")