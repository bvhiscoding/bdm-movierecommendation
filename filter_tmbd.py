import pandas as pd
import os

def filter_movielens_in_tmdb(movielens_files_dir, output_dir):
    """
    Lọc các phim MovieLens có tồn tại trong TMDB và lưu kết quả vào file mới
    Đồng thời lưu danh sách phim không tìm thấy trong TMDB vào file riêng
    
    Parameters:
    -----------
    movielens_files_dir : str
        Đường dẫn đến thư mục chứa các file dữ liệu MovieLens và TMDB
    output_dir : str
        Đường dẫn đến thư mục đầu ra để lưu file kết quả
    """
    print("Bắt đầu lọc phim MovieLens có trong TMDB...")
    
    # Tạo thư mục đầu ra nếu không tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục đầu ra: {output_dir}")
    
    # Tải dữ liệu phim từ MovieLens
    try:
        movies_file = os.path.join(movielens_files_dir, 'movie.csv')
        movies_df = pd.read_csv(movies_file)
        print(f"Đã tải dữ liệu phim MovieLens: {len(movies_df)} bộ phim")
        print(movies_df.head(3))
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu phim MovieLens: {str(e)}")
        return
    
    # Tải dữ liệu link để kết nối ID của MovieLens với TMDB
    try:
        links_file = os.path.join(movielens_files_dir, 'link.csv')
        links_df = pd.read_csv(links_file)
        print(f"Đã tải dữ liệu liên kết: {len(links_df)} bản ghi")
        print(links_df.head(3))
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu liên kết: {str(e)}")
        return
    
    # Tải dữ liệu TMDB
    try:
        tmdb_file = os.path.join(movielens_files_dir, 'tmdb.csv')
        tmdb_df = pd.read_csv(tmdb_file)
        print(f"Đã tải dữ liệu TMDB: {len(tmdb_df)} bộ phim")
        print(f"TMDB IDs đầu tiên: {tmdb_df['id'].head(5).tolist()}")
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu TMDB: {str(e)}")
        return
    
    # Chuyển đổi dữ liệu TMDB ID sang cùng kiểu với links_df
    tmdb_ids = set(tmdb_df['id'].astype(float).tolist())
    print(f"\nTổng số TMDB IDs: {len(tmdb_ids)}")
    
    # Ghép nối bảng phim MovieLens với bảng liên kết để có thông tin TMDB ID
    merged_df = pd.merge(movies_df, links_df, on='movieId', how='left')
    print(f"Kết quả sau khi ghép nối MovieLens với bảng liên kết: {len(merged_df)} bộ phim")
    
    # Chuyển đổi tmdbId sang float để dễ so sánh
    merged_df['tmdbId'] = merged_df['tmdbId'].astype(float)
    
    # Lọc các phim có trong TMDB
    movielens_in_tmdb = merged_df[merged_df['tmdbId'].isin(tmdb_ids)]
    print(f"Số phim MovieLens có trong TMDB: {len(movielens_in_tmdb)}")
    
    # Lọc các phim không có trong TMDB
    movielens_not_in_tmdb = merged_df[~merged_df['tmdbId'].isin(tmdb_ids)]
    print(f"Số phim MovieLens không có trong TMDB: {len(movielens_not_in_tmdb)}")
    
    # Lưu kết quả các phim có trong TMDB vào file CSV
    output_file = os.path.join(output_dir, 'tmdb_real.csv')
    # Chỉ lấy các thông tin từ MovieLens
    movielens_in_tmdb[['movieId', 'title', 'genres']].to_csv(output_file, index=False)
    print(f"\nĐã lưu kết quả các phim có trong TMDB vào: {output_file}")
    
    # Lưu kết quả các phim không có trong TMDB vào file CSV riêng
    missing_file = os.path.join(output_dir, 'notintmdb.csv')
    # Chỉ lấy các thông tin từ MovieLens
    movielens_in_tmdb[['movieId', 'title', 'genres']].to_csv(missing_file, index=False)
    print(f"Đã lưu kết quả các phim không có trong TMDB vào: {missing_file}")
    
    # Hiển thị thông tin tổng kết
    print(f"\nTổng kết:")
    print(f"- Tổng số phim MovieLens: {len(movies_df)}")
    print(f"- Số phim có trong TMDB: {len(movielens_in_tmdb)}")
    print(f"- Số phim không có trong TMDB: {len(movielens_not_in_tmdb)}")
    
    # Hiển thị một số phim có trong TMDB
    print("\nMẫu dữ liệu phim có trong TMDB:")
    print(movielens_in_tmdb[['movieId', 'title', 'genres']].head(3))
    
    # Hiển thị một số phim không có trong TMDB
    print("\nMẫu dữ liệu phim không có trong TMDB:")
    print(movielens_not_in_tmdb[['movieId', 'title', 'genres']].head(3))

# Đường dẫn thư mục
input_dir = './data/20m'  # Thay đổi theo cấu trúc thư mục của bạn
output_dir = './filtered_data'

# Gọi hàm chính
filter_movielens_in_tmdb(input_dir, output_dir)