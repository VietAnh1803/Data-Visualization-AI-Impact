# Phân tích dữ liệu Global AI Content Impact

Repo cung cấp phân tích khám phá có thể chạy lại cho [`Global_AI_Content_Impact_Dataset.csv`](data/Global_AI_Content_Impact_Dataset.csv). Đọc [báo cáo đầy đủ bằng tiếng Việt](reports/AI_impact_analysis_vi.md) để xem phương pháp, phát hiện, biểu đồ và giới hạn diễn giải.

## Kết quả chính

Mẫu gồm 200 dòng trong giai đoạn 2020–2025. Trung bình tỷ lệ ứng dụng AI là 54,27%. Tương quan Pearson giữa ứng dụng AI và tăng doanh thu được báo cáo gần bằng 0 (r = 0,002); không thể suy ra tác động nhân quả từ CSV này. Nguồn và phương pháp lấy mẫu chưa được cung cấp.

## Chạy lại

Yêu cầu Python 3.10+.

```bash
python -m pip install -r requirements.txt
python analyze_data.py
```

Script ghi bốn biểu đồ PNG và số liệu máy đọc được vào `reports/figures/`. Chạy được từ bất kỳ thư mục nào vì đường dẫn mặc định được tính theo vị trí script. Tùy chọn:

```bash
python analyze_data.py --data data/Global_AI_Content_Impact_Dataset.csv --output reports/figures
```

## Cấu trúc

- `analyze_data.py`: kiểm tra schema, số liệu mô tả, tương quan và biểu đồ.
- `data/`: CSV nguồn được giữ nguyên.
- `reports/AI_impact_analysis_vi.md`: báo cáo cho người đọc.
- `reports/figures/`: biểu đồ và `summary.json` có thể tái tạo.

Các PNG cũ ở thư mục gốc là đầu ra lịch sử của code trước đây. Báo cáo mới chỉ dẫn tới biểu đồ trong `reports/figures/`.
