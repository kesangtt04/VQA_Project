# VQA_Project

## Tóm tắt
Dự án xây dựng hệ thống Hỏi đáp hình ảnh (Visual Question Answering - VQA) sử dụng kỹ thuật học sâu. Mục tiêu là phát triển mô hình có khả năng nhận diện và trả lời câu hỏi dựa trên nội dung bức ảnh. Dự án tập trung xây dựng dữ liệu, thiết kế mô hình kết hợp CNN và LSTM, đồng thời thực hiện các thí nghiệm đánh giá và so sánh hiệu suất các phương pháp.

---

## Tổng quan

Hệ thống VQA nhận đầu vào là hình ảnh và câu hỏi, trả về câu trả lời tương ứng. Mô hình sử dụng kiến trúc đa phương thức kết hợp:

- **CNN** để trích xuất đặc trưng hình ảnh.
- **LSTM** để mã hóa câu hỏi thành biểu diễn ngữ nghĩa.
- Kết hợp đặc trưng để dự đoán câu trả lời chính xác.

---

## Xây dựng dữ liệu

- Sử dụng bộ dữ liệu Fruits-360 với hơn 80,000 ảnh trái cây và rau củ quả thuộc 160 lớp.
- Lọc chọn một số loại trái cây tiêu biểu, lấy 100 ảnh ngẫu nhiên mỗi lớp.
- Thay đổi kích thước ảnh về 64x64 để giảm tải tính toán.
- Câu hỏi đơn giản như “quả gì”, câu trả lời lấy từ nhãn ảnh.
- Dữ liệu lưu trong file `train_data.json` với cấu trúc:

```json
{
  "image_path": "path/to/image.jpg",
  "question": "quả gì",
  "answer": "quả táo"
}
