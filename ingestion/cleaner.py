import re
import unicodedata

def clean_data(raw_text: str) -> str:
    """
    Clean extracted document text before chunking.
    Optimized for production-grade RAG pipelines.
    """
    if not raw_text:
        return ""

    # 0. Chuẩn hóa Unicode (Ép về một dạng encode chuẩn để regex hoạt động chính xác)
    text = unicodedata.normalize("NFKC", raw_text)
    
    # 1. Nối từ bị ngắt dòng (Chỉ nối nếu giữa 2 chữ cái)
    # trans-\nformer -> transformer | high-\ntech -> high-tech (được giữ nguyên)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # 2. Xóa số trang / Page markers và các khoảng trắng xung quanh nó trên dòng đó
    # Tránh để lại các dòng trống vô nghĩa sau khi xóa
    text = re.sub(
        r'(?m)^[ \t]*(?:Trang|Page)\s+\d+\s*(?:/\s*\d+|of\s+\d+)?[ \t]*\n?',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # 3. Chuẩn hóa tất cả ký tự bullet lạ thành dấu '-' (Chạy 1 lần duy nhất bằng Regex)
    text = re.sub(r'[●■►▪▫•❑▪◦]', '-', text)
    
    # 4 & 5. Gộp Tabs và nhiều khoảng trắng liên tiếp thành 1 khoảng trắng đơn
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 6. Thu gọn các dòng trống thừa (Chỉ xét khoảng trắng ngang, không nuốt đoạn văn)
    # 3 hoặc nhiều dấu xuống dòng -> gộp thành 2 dấu xuống dòng (\n\n)
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    
    # 7. Trim đầu cuối văn bản
    text = text.strip()
    
    return text