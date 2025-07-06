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

Tiền xử lý:

Lọc và chọn lọc các loại trái cây tiêu biểu.

Giảm số lượng ảnh ngẫu nhiên xuống 100 ảnh mỗi lớp để cân bằng và giảm thời gian huấn luyện.

Thay đổi kích thước ảnh xuống 64x64 pixels.

Tạo các mẫu câu hỏi đơn giản (ví dụ: "đây là quả gì") và câu trả lời tương ứng từ nhãn ảnh (ví dụ: "quả táo").

Dữ liệu huấn luyện được lưu trữ dưới dạng JSON.

📈 Kết quả huấn luyện
Mô hình được huấn luyện với 

Adam optimizer và sparse_categorical_crossentropy loss, theo dõi accuracy.


Accuracy: Độ chính xác trên cả tập huấn luyện và tập kiểm định đều đạt mức rất cao (gần 1.0) sau một vài epochs.


Loss: Train Loss giảm nhanh chóng. Tuy nhiên, 

Validation Loss cho thấy dấu hiệu tăng trở lại sau một số epochs nhất định, cho thấy mô hình đang bị overfitting trên dữ liệu huấn luyện.


💡 Hướng phát triển trong tương lai
Để cải thiện khả năng tổng quát hóa của mô hình và khắc phục overfitting, các hướng phát triển tiềm năng bao gồm:

Áp dụng thêm các kỹ thuật Regularization (ví dụ: Dropout).

Tăng cường dữ liệu (Data Augmentation) để mở rộng bộ dữ liệu huấn luyện.

Khám phá các kiến trúc VQA tiên tiến hơn hoặc cơ chế Attention để kết hợp đặc trưng hiệu quả hơn.

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
