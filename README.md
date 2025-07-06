🚀 Giới thiệu dự án
Đây là một dự án Hệ thống Hỏi Đáp Hình Ảnh (VQA) chuyên biệt, tập trung vào việc nhận diện và trả lời các câu hỏi về trái cây. Hệ thống sử dụng các kỹ thuật Học Sâu để kết hợp thông tin từ hình ảnh và văn bản, nhằm cung cấp câu trả lời chính xác.

✨ Tính năng nổi bật
Nhận diện trái cây: Xác định loại trái cây trong ảnh đầu vào.

Hỏi đáp tự động: Trả lời các câu hỏi đơn giản liên quan đến trái cây trong ảnh.

Giao diện người dùng trực quan: Dễ dàng thêm dữ liệu huấn luyện mới và kiểm tra khả năng nhận diện.

🧠 Kiến trúc mô hình
Dự án được xây dựng trên kiến trúc đa phương thức, kết hợp sức mạnh của:

Image Feature Extractor (CNN): Sử dụng Convolutional Neural Networks để trích xuất các đặc trưng trực quan từ hình ảnh trái cây, tập trung vào hình dạng, màu sắc và kết cấu.

Question Encoder (LSTM): Sử dụng Long Short-Term Memory networks để mã hóa câu hỏi thành biểu diễn ngữ nghĩa, giúp mô hình hiểu ngữ cảnh.

Combination & Prediction: Các đặc trưng từ ảnh và câu hỏi được kết hợp (concatenation) và xử lý thông qua các lớp Dense, Dropout, và một lớp LSTM cuối cùng để tạo ra chuỗi câu trả lời.

📊 Bộ dữ liệu

Nguồn: Bộ dữ liệu Fruits-360.

💻 Cách sử dụng
Nhập dữ liệu huấn luyện (Data Input):

Mở giao diện "VQA System - Data Input".

Nhấn "Select Images" để chọn ảnh.

Nhập câu hỏi vào "Enter Question:" (ví dụ: "đây là quả gì").

Nhập câu trả lời vào "Enter Answer:" (ví dụ: "quả táo").

Nhấn "Save & Train" để lưu và bổ sung vào tập dữ liệu huấn luyện.

Nhận diện trái cây (Fruit Recognition - VQA):

Mở giao diện "Fruit Recognition - VQA".

Nhấn "chọn ảnh" để tải lên hình ảnh trái cây cần nhận diện.

Nhập câu hỏi vào ô "Nhập câu hỏi:" (ví dụ: "đây là quả gì").

Nhấn "nhận diện" và kết quả sẽ hiển thị ở phần "Result:".


| Giao diện huấn luyện                                                                           | Giao diện kiểm thử (Táo)                                                                  | Giao diện kiểm thử (Chuối)                                                                 |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| ![Data Input](https://github.com/user-attachments/assets/e9c22f91-aa80-4898-a470-1ea1398322bb) | ![Apple](https://github.com/user-attachments/assets/cfa93d37-94b2-4d71-8c90-2b0c69fdc537) | ![Banana](https://github.com/user-attachments/assets/fbb960b7-89e2-48ac-b3f0-767ad6b33c32) |



