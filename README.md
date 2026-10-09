# Phân tích dữ liệu Global AI Content Impact

Repo cung cấp phân tích khám phá có thể chạy lại cho [`Global_AI_Content_Impact_Dataset.csv`](data/Global_AI_Content_Impact_Dataset.csv). Mở [dashboard HTML tương tác](reports/dashboard.html) để lọc dữ liệu. Báo cáo học thuật có hai bản tương ứng: [tiếng Việt](reports/AI_impact_analysis_vi.md) và [English](reports/AI_impact_analysis_en.md).

## Kết quả chính

Mẫu gồm 200 dòng trong giai đoạn 2020–2025. Trung bình tỷ lệ ứng dụng AI là 54,27%. Tương quan Pearson giữa ứng dụng AI và tăng doanh thu được báo cáo gần bằng 0 (r = 0,002); không thể suy ra tác động nhân quả từ CSV này. Nguồn và phương pháp lấy mẫu chưa được cung cấp.

## Chạy lại

Yêu cầu Python 3.10+.

```bash
python -m pip install -r requirements.txt
python analyze_data.py
python build_dashboard.py
```

`analyze_data.py` ghi năm biểu đồ PNG bằng tiếng Anh, năm bản tiếng Việt và số liệu máy đọc được vào `reports/figures/`. `build_dashboard.py` nhúng CSV cùng các hình vào `reports/dashboard.html`; có thể mở file HTML trực tiếp bằng trình duyệt mà không cần server hay Internet. Năm hình có phần đọc số liệu và giới hạn ngay bên cạnh; bên dưới là các biểu đồ tương tác theo bộ lọc. Nút **VI / EN** đổi cả văn bản lẫn hình và giữ nguyên lựa chọn bộ lọc. Hai script chạy được từ bất kỳ thư mục nào vì đường dẫn mặc định được tính theo vị trí script. Tùy chọn cho phân tích:

```bash
python analyze_data.py --data data/Global_AI_Content_Impact_Dataset.csv --output reports/figures
```

## Cấu trúc

- `analyze_data.py`: kiểm tra schema, số liệu mô tả, tương quan và biểu đồ.
- `build_dashboard.py`: kiểm tra SHA-256 nguồn rồi nhúng dữ liệu và hình vào dashboard HTML.
- `data/`: CSV nguồn được giữ nguyên.
- `reports/AI_impact_analysis_vi.md` và `reports/AI_impact_analysis_en.md`: hai bản ngôn ngữ của cùng báo cáo.
- `reports/dashboard.html`: báo cáo trực quan offline với năm hình có diễn giải toàn mẫu và phần tương tác theo bộ lọc.
- `reports/figures/`: biểu đồ và `summary.json` có thể tái tạo.

Các PNG cũ ở thư mục gốc là đầu ra lịch sử của code trước đây. Báo cáo mới chỉ dẫn tới biểu đồ trong `reports/figures/`.
