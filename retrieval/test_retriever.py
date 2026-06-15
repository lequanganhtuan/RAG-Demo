import os
from pathlib import Path
import sys

# Import các hàm và cấu hình từ file của bạn
# (Giả sử file embedder.py và retriever.py nằm cùng thư mục)
ROOT_DIR = str(Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 2. Bây giờ import chắc chắn sẽ chạy được mà không lo lỗi ModuleNotFoundError
from embedder import build_faiss_index, FAISS_INDEX_PATH, METADATA_PATH, embed_chunks
from retriever import search_faiss

# ==========================================
# 1. TẠO DỮ LIỆU GIẢ LẬP (MÔ PHỎNG TẦNG CHUNKING)
# ==========================================
# Giả lập kết quả đầu ra sau khi chạy Parent-Child + Recursive Chunking
mock_parent_store = {
    "parent_0": (
        "Quy định về thời gian làm việc: Nhân viên văn phòng làm việc từ 8:00"
        " đến 17:00, từ thứ Hai đến thứ Sáu. Nghỉ trưa từ 12:00 đến 13:00. Việc"
        " đi muộn quá 3 lần một tháng không có lý do chính đáng sẽ bị nhắc nhở"
        " bằng văn bản."
    ),
    "parent_1": (
        "Quy định về nghỉ phép và nghỉ bệnh: Nhân viên cần nộp đơn xin nghỉ"
        " phép trước ít nhất 3 ngày làm việc. Đối với trường hợp nghỉ bệnh đột"
        " xuất, nhân viên phải thông báo cho quản lý trực tiếp trước 9:00 sáng"
        " và bổ sung giấy xác nhận của bác sĩ trong vòng 24 giờ kể từ khi đi"
        " làm lại."
    ),
    "parent_2": (
        "Chính sách bảo mật thông tin: Tất cả tài liệu dự án, mã nguồn và dữ"
        " liệu khách hàng là tài sản tối mật của công ty. Nhân viên tuyệt đối"
        " không được sao chép, chia sẻ ra bên ngoài hệ sinh thái của công ty"
        " dưới mọi hình thức nếu chưa có sự phê duyệt từ Giám đốc."
    ),
}

mock_vector_database = [
    # Các mảnh con của Parent 0 (Giờ làm việc)
    {
        "child_id": "parent_0_child_0",
        "parent_id": "parent_0",
        "text": "Nhân viên văn phòng làm việc từ 8:00 đến 17:00, từ thứ Hai đến thứ Sáu.",
        "vector": None,
    },
    {
        "child_id": "parent_0_child_1",
        "parent_id": "parent_0",
        "text": "Nghỉ trưa từ 12:00 đến 13:00. Đi muộn quá 3 lần/tháng sẽ bị nhắc nhở bằng văn bản.",
        "vector": None,
    },
    # Các mảnh con của Parent 1 (Nghỉ phép/Nghỉ bệnh)
    {
        "child_id": "parent_1_child_0",
        "parent_id": "parent_1",
        "text": "Nhân viên cần nộp đơn xin nghỉ phép trước ít nhất 3 ngày làm việc.",
        "vector": None,
    },
    {
        "child_id": "parent_1_child_1",
        "parent_id": "parent_1",
        "text": "Nghỉ bệnh đột xuất phải báo trước 9:00 sáng và nộp giấy bác sĩ trong 24 giờ.",
        "vector": None,
    },
    # Mảnh con của Parent 2 (Bảo mật)
    {
        "child_id": "parent_2_child_0",
        "parent_id": "parent_2",
        "text": "Tài liệu dự án, mã nguồn và dữ liệu khách hàng là tài sản tối mật của công ty.",
        "vector": None,
    },
]


def run_pipeline_test():
    print("=== BƯỚC 1: TIẾN HÀNH EMBED CHUNKS VÀ LƯU METADATA ===")
    updated_db, embeddings_matrix = embed_chunks(mock_vector_database)
    print(f"-> Đã tạo xong {len(embeddings_matrix)} vectors.")
    print(f"-> File metadata đã lưu tại: {METADATA_PATH}\n")

    print("=== BƯỚC 2: BUILD FAISS INDEX (FULL REBUILD) ===")
    index = build_faiss_index(embeddings_matrix)
    print(f"-> Đã nạp {index.ntotal} vectors vào FAISS.")
    print(f"-> File FAISS index đã lưu tại: {FAISS_INDEX_PATH}\n")

    print("=== BƯỚC 3: KIỂM TRA TRUY VẤN (RETRIEVER TEST) ===")

    # Test case 1: Câu hỏi đúng trọng tâm tài liệu nghỉ bệnh
    query_1 = "Bị ốm đột xuất thì phải nộp giấy tờ gì và hạn chót là khi nào?"
    print(f"\n[Query 1]: '{query_1}'")

    results_1 = search_faiss(
        query_text=query_1, parent_store=mock_parent_store, top_k=2
    )

    print("\n--- Chi tiết các mảnh Con tìm được (child_details) ---")
    child_items = results_1["child_details"]
    for item in child_items:
        print(
            f"Mã con: {item['child_id']} | Điểm Cosine: {item['score']:.4f}"
        )
        print(f"Nội dung con: {item['text']}\n")

    print("--- Ngữ cảnh Cha trọn vẹn trả về cho LLM (context_for_llm) ---")
    print(results_1["context_for_llm"])
    print("-" * 60)

    # Test case 2: Câu hỏi nằm ngoài tài liệu (Để test Relevance Threshold)
    # Vì bạn đã cài đặt ngưỡng chặn score (ví dụ > 0.3), câu này sẽ trả về context rỗng
    query_2 = "Hướng dẫn cách nấu món phở bò Nam Định ngon chuẩn vị"
    print(f"\n[Query 2]: '{query_2}'")

    results_2 = search_faiss(
        query_text=query_2, parent_store=mock_parent_store, top_k=2
    )

    print("\n--- Kết quả lọc nhiễu ngoài tài liệu ---")
    if not results_2["child_details"]:
        print(
            " Thử nghiệm thành công: Hệ thống đã chặn câu hỏi ngoài lề (Score"
            " không vượt qua ngưỡng tối thiểu)!"
        )
    else:
        print(
            "Cảnh báo: Vẫn có mảnh lọt qua, kiểm tra lại ngưỡng threshold"
            " trong retriever.py"
        )


if __name__ == "__main__":
    # Đảm bảo dọn dẹp file cũ trước khi chạy test
    if FAISS_INDEX_PATH.exists():
        os.remove(FAISS_INDEX_PATH)
    if METADATA_PATH.exists():
        os.remove(METADATA_PATH)

    run_pipeline_test()