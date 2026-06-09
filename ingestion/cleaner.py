from loader import load_document
import re

raw_text = load_document()

def clean_data(raw_text: str) -> str:
    """
    Clean extracted document text before chunking.
    """
    # 1. Join hyphenated words across line breaks
    # trans-\nformer -> transformer
    cleam_text = re.sub(r'-\s*\n\s*', '', raw_text)
    
    # 2. Remove common page markers
    # Page 5 of 20,...
    cleam_text = re.sub(
        r'(Trang|Page)\s+\d+\s*(?:/\s*\d+|of\s+\d+)?',
        '',
        cleam_text,
        flags=re.IGNORECASE
    )
    
    # 3. Normalize bullet characters
    bullet_chars = ['●', '■', '►', '▪', '▫', '•']

    for bullet in bullet_chars:
        cleam_text = cleam_text.replace(bullet, '-')

    # 4. Replace tabs with spaces
    cleam_text = cleam_text.replace('\t', ' ')

    # 5. Collapse multiple spaces
    cleam_text = re.sub(r' +', ' ', cleam_text)

    # 6. Remove excessive blank lines
    cleam_text = re.sub(r'\n\s*\n+', '\n\n', cleam_text)

    # 7. Trim leading/trailing whitespace
    cleam_text = cleam_text.strip()

    return cleam_text