import os
import logging
from typing import List, Dict
from pypdf import PdfReader

class PdfParser:
    """
    A robust class to handle PDF parsing and text extraction from individual or multiple PDF files.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extracts text from each page of a PDF and returns it with basic metadata.

        Args:
            pdf_path (str): The full path to the PDF file.

        Returns:
            List[Dict]: Each dict contains 'page_content', 'page_number', and 'file_path'.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        self.logger.info(f"Extracting text from: {pdf_path}")
        extracted_data = []

        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_data.append({
                        'page_content': page_text.strip(),
                        'page_number': i + 1,
                        'file_path': pdf_path
                    })
            self.logger.info(f"Extracted {len(extracted_data)} pages from: {pdf_path}")
        except Exception as e:
            self.logger.warning(f"Error extracting text from {pdf_path}: {e}")
            return []

        return extracted_data

    def extract_from_folder(self, folder_path: str) -> List[Dict]:
        """
        Extracts text from all PDFs within a given folder.

        Args:
            folder_path (str): Path to the directory containing PDF files.

        Returns:
            List[Dict]: Combined list of page-level dictionaries from all PDFs.
        """
        all_pages = []
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Provided path is not a directory: {folder_path}")

        self.logger.info(f"Scanning folder: {folder_path}")

        for file in os.listdir(folder_path):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                self.logger.info(f"Processing file: {pdf_path}")
                all_pages.extend(self.extract_text_from_pdf(pdf_path))

        self.logger.info(f"Total pages extracted from folder: {len(all_pages)}")
        return all_pages
