---
layout: post
title: "Chương 1: Nền tảng của Mạng Nơ-ron Nhân tạo"
date: 2025-10-03 09:00 +0700
categories: [Deep Learning, Notes]
tags: [ANN, PyTorch, Feedforward, BackPropagation]
permalink: /posts/chuong-1-ann/
toc: true  
---
_Xin chào mọi người, một số chủ đề mình sẽ chia sẻ và cùng nhau học với mọi người sẽ là AI - Computer Vision - Mong mọi người cùng nhau học hỏi và cùng nhau phát triển...._
## Khái niệm
**Mạng Nơ-Ron nhân tạo ANN (Artificial Neural Network)** là một thuật toán học có giám sát, lấy cảm hứng từ cách hoạt động của não người. Mạng nhận **đầu vào** → **xử lý qua hàm kích hoạt** → **kích hoạt nơ-ron tiếp theo** → **tạo đầu ra (output)**.

## Đặc điểm chính: 
- Có nhiều kiến trúc mạng chuẩn: **MLP, CNN, RNN...**
- **Định lý xấp xỉ phổ quát (Universal Approximation Theorem):** Một mạng nơ-ron đủ lớn, được huấn luyện đúng cách có thể mô phỏng **bất kỳ hàm nào**, từ đó dự đoán output từ input tương ứng.  
- **Huấn luyện mạng nơ-ron:** Với tập dữ liệu cụ thể, ta xây dựng một kiến trúc mạng, điều chỉnh trọng số cho đến khi mô hình dự đoán kết quả mong muốn.

## Ứng dụng điển hình:
- Trong **thị giác máy tính**, ANN được dùng để phân loại ảnh.  
- **Cuộc thi ImageNet** là cột mốc quan trọng: các mô hình ANN thi nhau phân loại hàng triệu ảnh, tỷ lệ lỗi giảm mạnh theo từng năm.
  
![Tỷ lệ lỗi ImageNet](/assets/img/chuong1/figure1.png)

*Hình 1: Tỷ lệ lỗi phân loại trong cuộc thi ImageNet.*

- Năm 2012, **AlexNet** chiến thắng ImageNet. Tỷ lệ lỗi giảm mạnh, thậm chí tốt hơn con người.
- Ý nghĩa: Mạng nơ-ron sâu không chỉ phân loại ảnh tốt mà còn mở ra nhiều ứng dụng mới.

## Ứng dụng mở rộng (Generative AI)
- Sinh ảnh từ văn bản đầu vào.  
- Sinh ảnh mới từ ảnh + văn bản.  
- Kết hợp đa dạng dữ liệu (ảnh, văn bản, âm thanh) để sinh nội dung.  
- Sinh video từ văn bản hoặc ảnh.  

💡 Đây chính là động lực quan trọng để học và triển khai ANN cho các bài toán tùy chỉnh.

## Nội dung chính của chương 1
- So sánh **AI** và **học máy truyền thống**.  
- Hiểu các khối cơ bản của ANN: **feedforward, backpropagation, learning rate**.  
- Triển khai **feedforward propagation**.  
- Triển khai **backpropagation**.  
- Kết hợp cả hai để huấn luyện.  
- Phân tích tác động của **learning rate**.  
- Tóm tắt quá trình huấn luyện mạng nơ-ron.

## I. So sánh trí tuệ nhân tạo (AI) và học máy truyền thống

Trước đây, cách tiếp cận để làm hệ thống “thông minh” là việc các lập tình viên phải viết ra thuật toán phức tạp dựa trên kiến thức từ chuyên gia. Ví dụ về việc đang quan tâm đén việc nhận diện xem một bức ảnh có chứa hình con chó hay không.
Trong thiết lập học máy truyền thống (Machine Learning) thì một chuyên gia ML hoặc một chuyên gia lĩnh vực sẽ: **xác định các đặc trưng (features) cần được trích xuất từ hình ảnh -> Trích xuất các đặc trưng đó -> Đưa vào thuật toán phân loại (SVM, Random Forest, KNN…) -> quyết định ảnh có là chó không.**

![Traditional Machine Learning workflow for classification](/assets/img/chuong1/figure2.png)

*Hình 2: Traditional Machine Learning workflow for classification*

![Sample images to generate rules](/assets/img/chuong1/figure3.png)

*Hình 3: Sample images to generate rules* 

Từ các ảnh trên, một quy tắc đơn giản có thể hiểu: nếu một bức ảnh chứa 3 vùng tròn màu đen được sắp xếp theo hình tam giác, thì ảnh đó có thể được phân loại là chó. Tuy nhiên nó lại là bánh muffin.

Hạn chế của phương pháp truyền thống: 
- **Quy tắc thủ công**: ví dụ đặt ra quy tắc “ba vòng tròn đen thành hình tam giác -> chó”.
- **Vấn đề**:
- Dễ bị đánh lừa (ảnh bánh muffin giống mặt chó).
- Cần quá nhiều quy tắc khi hình ảnh phức tạp, thay đổi lớn.
- Chỉ hiệu quả trong môi trường ràng buộc chặt chẽ (ví dụ: ảnh hộ chiếu)
- Khó áp dụng cho các tình huống đa dạng, thực tế.

**Lợi thế của mạng Nơ-ron nhân tạo (ANNs):**
- Kết hợp hai bước: tự động trích xuất đặc trưng và phân loại trong một lần.
- Ít công sức thủ công: không cần con người định nghĩa quy tắc phức tạp.
- Điều kiện tiên quyết: cần tập dữ liệu gán nhãn đủ lớn 

**Cách tiếp cận dựa trên ANN cho phân loại:**
- Input -> Mạng học Features -> Dùng chính đặc trưng đó để phân loại.
- Toàn bộ quy trình được tối ưu tự động qua huấn luyện, thay vì viết quy tắc bằng tay.
  
![Neural network based approach for classification](/assets/img/chuong1/figure4.png)

*Hình 4: Neural network based approach for classification* 


## II. Hiểu các khối cơ bản của ANN: feedforward, backpropagation, learning rate

**1. Bản chất:**
- ANN là một hàm toán học gồm các trọng số (weights) và phép toán sắp xếp thành một kiến trúc mạng.
- Mạng nhận tensor đầu vào → xử lý qua các lớp → xuất tensor đầu ra.
- Kiến trúc mạng phụ thuộc loại dữ liệu: có cấu trúc (bảng, tabular) hay phi cấu trúc (ảnh, văn bản, âm thanh).

**2. Các lớp chính trong ANN**
- **Input layer**: nhận dữ liệu gốc (ví dụ: pixel ảnh, đặc trưng từ bảng).
- **Hideen Layer**:
  - Biến đổi dữ liệu từ đầu vào
  - Gồm nhiều nút (nodes), mỗi nút thực hiện tính toán.
  - Sử dụng hàm kích hoạt (activation function) để tạo phi tuyến, giúp mạng biểu diễn được mối quan hệ phức tạp.
- **Output Layer**: đưa ra kết quả cuối cùng (ví dụ: phân loại ảnh mèo/ chó, dự đoán giá trị số).
- **Tóm lại - ANN** gồm: input -> hidden layer -> output, với ưu điểm từ hidden layer và hàm kích hoạt giúp mạng học được các đặc trưng phức tạp mà cách thủ công không làm được.

![Neural network structure](/assets/img/chuong1/figure5.png)

*Hình 5: Neural network structureNeural network structure* 

Số lượng nút (các vòng tròn ở hình trên) ở output layer phụ thuộc vào bài toán cụ thể và việc chúng ta đang ở bài toán nào (dự đoán một biến liên tục hay phân loại?):
- **Bài toán hồi quy (regression)**: dự doán giá trị liên tục -> lớp đầu ra chỉ có 1 nút.
- **Bài toán phân loại (classification)**: dự đoán trong m lớp -> lớp đầu ra có m nút, mỗi nút ứng với một lớp.

Phóng to vào **một nút/ nơ-ron**. Một nơ-ron sẽ biến đổi input của nó theo cách sau: 

![Input transformation at a neuron](/assets/img/chuong1/figure6.png)

*Hình 6: Input transformation at a neuron* 

Các kí hiệu trên hình: 
- **Đầu vào x_1, x_2, x_n** : các biến đầu vào
- **W_0: là hệ số chệch (bias term)** (giúp mô hình linh hoạt, giống trong bài toán hồi quy tuyến tính hoặc logistic).
- **W_1, W_2, W_3,… W_n**: là các trọng số gán cho từng biến đầu vào, và W_0 là hệ số chệch.
- **Giá trị đầu ra (output) a** được tính như sau:
$$
\\( a = f(w_0 + \sum_{i=1}^n w_i x_i) \\)
$$
- Trong đó, **f là hàm kích hoạt (activation function)**
Nghĩa là: **đầu vào (x_1, x_2…x_n) × trọng số (W_1, W_2, W_3,… W_n) + bias (W_0) → qua hàm kích hoạt (f) → ra đầu ra  - output (a) của nơ-ron**.
**Ý nghĩa của hàm kích hoạt.**
- Tạo tính phi tuyến, giúp mạng học được các quan hệ phức tạp (nếu chỉ tuyến tính thì mạng giống hệt hồi quy tuyến tính)
- Có nhiều loại hàm kích hoạt (sigmoid, tanh, ReLU…) sẽ học kỹ hơn trong phần feedforward.
- **Khi một mạng nơ ron có nhiều lớp ẩn (hidden layers) thì ta gọi nó là deep learning.**
- **Nhiệm vụ càng phức tạp -> cần nhiều lớp ẩn hơn để xử lý - ví dụ như nhận diện hình ảnh.**

## III. Implementing feedforward propagation
- Để xây dựng nền tảng vững chắc về cách hoạt động của lan truyền tiến (feedforward propagation), ta sẽ xét ví dụ cơ bản sau về việc huấn luyện mạng nơ-ron. 
- Trong ví dụ này, đầu vào của mạng nơ-ron là (1,1) và đầu ra tương ứng mong đợi là (0). Ở đây, ta sẽ tìm các trọng số tối ưu của mạng nơ-ron dựa trên cặp dữ liệu input và output đã cho.
- Kiến trúc mạng nơ-ron của chúng ta trong ví dụ này bao gồm một lớp ẩn với ba nút như sau: 

![Sample neural network architecture with 1 hidden layer](/assets/img/chuong1/figure7.png)

*Hình 7: Sample neural network architecture with 1 hidden layer* 

- Xét hình 7:
  - **Mỗi mũi tên = một trọng số (float)** có thể điều chỉnh.
  - Có 6 trọng số từ 2 nút đầu vào -> 3 nút ẩn.
  - Có 3 trọng số từ 3 nút ẩn -> 1 nút đầu ra.
  - Tổng cộng 9 trọng số cần học.
- **Mục tiêu huấn luyện: tìm ra giá trị của 9 trọng số này sao cho: Input = (1,1) => Output = 0.**
- Ở đây chưa xét bias để giữ cho ví dụ đơn giản, logic vẫn đúng.
Các bước tiếp theo cần tìm hiểu:
- Tính toán lớp ẩn: đầu vào nhân với trọng số → ra giá trị trung gian.
- Hàm kích hoạt phi tuyến: biến đổi các giá trị trung gian để mạng học được quan hệ phức tạp.
- Ước lượng lớp đầu ra: từ lớp ẩn → đầu ra dự đoán.
- Hàm mất mát (Loss function): so sánh đầu ra dự đoán với đầu ra mong đợi (expected output).

**TÍNH TOÁN TRỌNG SỐ CHO VÍ DỤ HÌNH 7**
- Gán trọng số cho tất cả các kết nối (thông thường, các mạng nơ-ron được khởi tạo bằng các trọng số ngẫu nhiên trước khi quá trình huấn luyện bắt đầu).
-	Ban đầu – bước khởi tạo - các trọng số thường được gán ngẫu nhiên (ví dụ trong khoảng [0,1].
-	Trong ví dụ này, ta bỏ qua bias để tập trung vào cơ chế feedforward và backpropagation.

![GanTrongSo](/assets/img/chuong1/figure8.png)

*Hình 8: Gán trọng số & khởi tạo ngẫu nhiên* 

- Các trọng số và giá trị trong mạng được thể hiện ở sơ đồ sau (nửa bên trái), và các trọng số khởi tạo ngẫu nhiên được minh họa trong mạng ở nửa bên phải.
- Ở bước tiếp theo, ta thực hiện phép nhân giữa input và các trọng số để tính toán các giá trị của các đơn vị trong lớp ẩn. Các giá trị của các đơn vị trong lớp ẩn trước khi áp dụng hàm kích hoạt (activation) được xác định như sau: 

$$
h_11= X_1× W_11+ X_2×W_21=1×0.8+1 ×0.2=1
$$
$$
h_12= X_1× W_12+ X_2×W_22=1×0.4+1 ×0.9=1.3
$$
$$
h_13= X_1× W_13+ X_2×W_23=1×0.3+1 ×0.5=0.8
$$

- Kết quả của các đơn vị trong hidden layer (trước khi áp dụng hàm kích hoạt) sau khi tính toán xong ở trên:

![TinhToan](/assets/img/chuong1/figure9.png)

*Hình 9: Kết quả cuối cùng trước khi áp dụng hàm kích hoạt* 





