import re
import logging
from typing import List, Dict

class ScientificTextPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.info("Initialized ScientificTextPreprocessor")

    def preprocess(self, raw_text: str) -> List[Dict]:
        blocks = []
        lines = raw_text.splitlines()
        buffer = []

        figure_caption_start_pattern = re.compile(r'^(?:Fig\.|Figure)\s+\d+[\.:]?\s+')
        table_caption_start_pattern = re.compile(r'^Table\s+\d+[\.:]?\s+')
        appendix_pattern = re.compile(r'^Appendix(?:\s+[A-Z])?(?:[\.:]\s+.+)?$', re.IGNORECASE)
        references_section_pattern = re.compile(r'^References\s*$', re.IGNORECASE)

        is_caption = False
        caption_type = None
        caption_buffer = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if references_section_pattern.match(line):
                if buffer:
                    blocks.append({"type": "paragraph", "content": "\n".join(buffer)})
                    buffer = []
                blocks.append({"type": "references", "content": ""})
                continue

            if figure_caption_start_pattern.match(line):
                if buffer:
                    blocks.append({"type": "paragraph", "content": "\n".join(buffer)})
                    buffer = []
                if is_caption:
                    blocks.append({"type": caption_type, "content": "\n".join(caption_buffer)})
                is_caption = True
                caption_type = "figure_caption"
                caption_buffer = [line]
                continue

            if table_caption_start_pattern.match(line):
                if buffer:
                    blocks.append({"type": "paragraph", "content": "\n".join(buffer)})
                    buffer = []
                if is_caption:
                    blocks.append({"type": caption_type, "content": "\n".join(caption_buffer)})
                is_caption = True
                caption_type = "table_caption"
                caption_buffer = [line]
                continue

            if appendix_pattern.match(line):
                if buffer:
                    blocks.append({"type": "paragraph", "content": "\n".join(buffer)})
                    buffer = []
                if is_caption:
                    blocks.append({"type": caption_type, "content": "\n".join(caption_buffer)})
                    caption_buffer = []
                    is_caption = False
                blocks.append({"type": "appendix", "section_title": line, "content": ""})
                continue

            if is_caption:
                caption_buffer.append(line)
                continue

            buffer.append(line)

        if is_caption:
            blocks.append({"type": caption_type, "content": "\n".join(caption_buffer)})

        if buffer:
            blocks.append({"type": "paragraph", "content": "\n".join(buffer)})

        return blocks