# Phân tích mô tả các chỉ số liên quan đến AI trong bộ dữ liệu Global AI Content Impact

[English version](AI_impact_analysis_en.md) · [Dashboard tương tác](dashboard.html)

**Dữ liệu:** [`Global_AI_Content_Impact_Dataset.csv`](../data/Global_AI_Content_Impact_Dataset.csv), 200 bản ghi, giai đoạn 2020–2025. **Ngày phân tích:** 09/10/2026. **Dấu vân tay SHA-256 của CSV:** `53b52d3d5fef30db9f575852ddcf6cbe347e049aef4de145f689f456fbf7a514`.

## Tóm tắt

Báo cáo mô tả phân bố của các chỉ số liên quan đến AI và xem xét mối liên hệ tuyến tính giữa chúng trong tập dữ liệu hiện có. Tỷ lệ ứng dụng AI trung bình của 200 bản ghi là 54,27%; tỷ lệ tăng doanh thu và mất việc được gắn nhãn “do AI” lần lượt là 39,72% và 25,79%. Tương quan Pearson giữa ứng dụng AI và tăng doanh thu gần bằng không (r = 0,002; khoảng tin cậy 95%: −0,137 đến 0,141). Trong 21 cặp chỉ số định lượng, không cặp nào đạt ngưỡng q < 0,05 sau hiệu chỉnh Benjamini–Hochberg. Những kết quả này mô tả các dòng trong CSV; nguồn thu thập, thiết kế mẫu và định nghĩa vận hành của chỉ số chưa được cung cấp. Vì vậy, báo cáo không ước lượng tác động nhân quả hoặc tỷ lệ đại diện cho một tổng thể rộng hơn.

## 1. Dữ liệu và phạm vi phân tích

CSV gồm 12 cột: quốc gia, năm, ngành, bảy chỉ số định lượng và hai biến phân loại về công cụ AI được dùng nhiều nhất và tình trạng quy định. Bảy chỉ số định lượng gồm sáu tỷ lệ phần trăm — ứng dụng AI, mất việc, tăng doanh thu, hợp tác người–AI, niềm tin người tiêu dùng và thị phần doanh nghiệp AI — cùng khối lượng nội dung do AI tạo (TB/năm). Dữ liệu trải trên 10 quốc gia, 10 ngành và sáu năm từ 2020 đến 2025.

Không có ô trống hoặc dòng trùng hoàn toàn. Các cột phần trăm nằm trong khoảng 0–100 và khối lượng nội dung đều dương. Tuy nhiên, **28 bản ghi lặp tổ hợp quốc gia–năm–ngành** so với một bản ghi trước đó; một tổ hợp có thể xuất hiện tới bốn lần. CSV không có mã định danh đơn vị quan sát, nên không có cơ sở xác định các dòng này là quan sát độc lập hay lỗi ghi nhận. Phân tích giữ nguyên toàn bộ 200 dòng.

Repo cũng không cung cấp đơn vị thu thập, quy trình chọn mẫu, công thức tính chỉ số hoặc cách xác định phần tăng doanh thu và mất việc “do AI”. Tên cột cho biết nhãn của biến, nhưng chưa đủ để kiểm chứng phép đo. Tỷ lệ trung bình trong báo cáo vì thế là **trung bình của các bản ghi**, không phải tỷ lệ của toàn bộ quốc gia, ngành hay thị trường toàn cầu.

## 2. Phương pháp

Các thống kê mô tả gồm trung bình, trung vị và độ lệch chuẩn mẫu. Mọi dòng có trọng số bằng nhau; trung bình theo năm không được chuẩn hóa theo cơ cấu quốc gia hoặc ngành. Hồi quy tuyến tính đơn biến của chỉ số theo năm được dùng để mô tả độ dốc trong mẫu. Khoảng 95% quanh trung bình năm được tính bằng phân phối t.

Quan hệ giữa bảy chỉ số định lượng được mô tả bằng hệ số Pearson r; năm không được đưa vào ma trận tương quan chỉ số. Có 21 cặp so sánh. Khoảng tin cậy 95% cho r dùng phép biến đổi Fisher, còn p-value của 21 cặp được hiệu chỉnh bằng phương pháp Benjamini–Hochberg. Những khoảng tin cậy và phép kiểm định này dựa trên giả định thống kê về tính độc lập và cơ chế lấy mẫu. Tài liệu nguồn chưa cho phép xác nhận các giả định đó; chúng được trình bày để người đọc thấy độ bất định của ước lượng, không để suy rộng kết quả ra ngoài mẫu.

## 3. Kết quả

### 3.1. Phân bố các chỉ số chính

![Phân bố ứng dụng AI, mất việc và tăng doanh thu](figures/01_distributions.png)

*Hình 1. Phân bố ba tỷ lệ được báo cáo; vạch đứng biểu thị trung bình mẫu. Nguồn: CSV của repo, 2020–2025, n = 200.*

Ứng dụng AI có trung bình 54,27%, trung vị 53,31% và độ lệch chuẩn 24,22 điểm phần trăm. Các giá trị tương ứng của tăng doanh thu là 39,72%, 42,10% và 23,83 điểm phần trăm; của mất việc là 25,79%, 25,74% và 13,90 điểm phần trăm. Mức phân tán này cần được lưu ý khi đọc một trị số trung bình duy nhất.

### 3.2. Khác biệt giữa các năm

![Trung bình ứng dụng AI và tăng doanh thu theo năm](figures/02_year_means.png)

*Hình 2. Trung bình theo năm với khoảng 95% tính bằng phân phối t. Cỡ mẫu lần lượt từ 2020 đến 2025 là 47, 32, 31, 29, 23 và 38 dòng. Nguồn: CSV của repo.*

Tỷ lệ ứng dụng AI trung bình là 50,99% vào năm 2020, đạt 59,68% vào năm 2023 và ở mức 54,26% vào năm 2025. Hồi quy đơn biến cho độ dốc 0,33 điểm phần trăm mỗi năm (p = 0,725). Với tăng doanh thu, độ dốc là −1,14 điểm phần trăm mỗi năm (p = 0,221). Cả hai chuỗi đều dao động qua các năm. Số dòng và thành phần mẫu thay đổi theo năm; dữ liệu không xác nhận rằng cùng một đơn vị được theo dõi liên tục.

### 3.3. Mối liên hệ giữa các chỉ số

![Ma trận tương quan của bảy chỉ số định lượng](figures/03_correlations.png)

*Hình 3. Hệ số Pearson r tính trên bảy chỉ số định lượng, n = 200. Màu thể hiện dấu và độ lớn của tương quan. Nguồn: CSV của repo.*

![Ứng dụng AI và tăng doanh thu được báo cáo](figures/04_adoption_revenue.png)

*Hình 4. Mỗi điểm tương ứng một bản ghi; đường thẳng là hồi quy tuyến tính mô tả. Nguồn: CSV của repo, n = 200.*

Giữa ứng dụng AI và tăng doanh thu, r = 0,002 (p = 0,979; khoảng tin cậy 95%: −0,137 đến 0,141). Giữa ứng dụng AI và mất việc, r = −0,005 (p = 0,949). Trong 21 cặp, trị tuyệt đối lớn nhất thuộc cặp mất việc–tăng doanh thu (r = 0,153; p chưa hiệu chỉnh = 0,031). Sau hiệu chỉnh nhiều phép thử, cặp này có q = 0,644. Không có cặp nào đạt q < 0,05. Kết quả cho thấy dữ liệu hiện tại không ghi nhận quan hệ tuyến tính rõ về độ lớn giữa các chỉ số chính; chúng không chứng minh rằng tác động thực tế bằng không.

### 3.4. Một so sánh giữa các ngành

Trong 10 ngành, Gaming có ứng dụng AI trung bình cao nhất (60,42%; n = 27) nhưng tăng doanh thu được báo cáo thấp nhất (33,23%). Media có ứng dụng AI trung bình thấp nhất (47,26%; n = 31), trong khi tăng doanh thu đạt 43,72%. Sự đảo chiều thứ hạng này đáng để khảo sát thêm, nhưng không cho biết nguyên nhân. Các trung bình ngành chưa được điều chỉnh theo quốc gia, năm hoặc đặc điểm đơn vị quan sát; báo cáo cũng không thực hiện phép kiểm định chênh lệch giữa hai ngành.

## 4. Diễn giải và giới hạn

Các nhãn “Due to AI” trong CSV là mô tả do nguồn dữ liệu đặt ra. Không có thiết kế đối chứng, dữ liệu trước–sau trên cùng đơn vị, hoặc thông tin về các yếu tố gây nhiễu để kiểm tra cách quy thuộc kết quả cho AI. Vì vậy, các hệ số trong báo cáo được diễn giải như mối liên hệ giữa các biến được ghi nhận.

Thiếu thiết kế mẫu và trọng số cũng giới hạn khả năng suy rộng. Số bản ghi khác nhau giữa các năm, quốc gia và ngành; 28 dòng lặp tổ hợp quốc gia–năm–ngành làm cho tính độc lập của quan sát chưa rõ. Các khoảng tin cậy và p-value chỉ có thể được hiểu theo đúng giả định thống kê đã nêu ở mục 2. Việc không đạt ngưỡng kiểm định không phải bằng chứng rằng AI không có ảnh hưởng.

Trước khi dùng dữ liệu để ra quyết định, cần có tài liệu về đơn vị quan sát, nguồn thu thập, công thức và thời điểm đo từng chỉ số, quy tắc xử lý bản ghi lặp và quyền sử dụng dữ liệu. Nếu mục tiêu là đánh giá tác động, bước tiếp theo là thu thập dữ liệu theo cùng đơn vị qua thời gian và xác định một thiết kế so sánh phù hợp.

## 5. Tái lập kết quả

Từ thư mục gốc của repo:

```bash
python -m pip install -r requirements.txt
python analyze_data.py
python build_dashboard.py
```

Script đầu tiên tạo bốn hình và [`summary.json`](figures/summary.json) trong `reports/figures/`. Script thứ hai nhúng CSV hiện tại vào [dashboard HTML](dashboard.html) để mở offline. `analyze_data.py` nhận tùy chọn `--data` và `--output` khi cần thay nguồn hoặc thư mục đầu ra. Các hình PNG cũ ở thư mục gốc là sản phẩm của phiên bản code trước, không được dùng để lập báo cáo này.
