import pandas as pd

def save_first_100_rows_to_csv(input_filename, output_filename):
    # Đọc toàn bộ dữ liệu từ file CSV vào DataFrame
    df = pd.read_csv(input_filename)
    # Lấy 100 dòng đầu tiên
    first_100 = df.head(10)
    # Ghi DataFrame chứa 100 dòng đầu tiên thành file CSV mới, không lưu cột chỉ mục
    first_100.to_csv(output_filename, index=False)

if __name__ == "__main__":
    input_file = "./data/20m/tag.csv"           # Đường dẫn đến file CSV gốc
    output_file = "./sample/tag_sample.csv"       # Tên file CSV mới
    save_first_100_rows_to_csv(input_file, output_file)
    print("Đã lưu 10 dòng đầu tiên vào file:", output_file)
