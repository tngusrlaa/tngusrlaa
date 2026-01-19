import os
import struct
import csv
from collections import defaultdict

JPEG_MARKERS = {
    0xFFD8: "SOI", 0xFFD9: "EOI", 0xFFDA: "SOS",
    0xFFC0: "SOF0", 0xFFC4: "DHT", 0xFFDB: "DQT",
    0xFFE0: "APP0", 0xFFE1: "APP1", 0xFFE2: "APP2",
    0xFFE3: "APP3", 0xFFE4: "APP4", 0xFFE5: "APP5",
    0xFFE6: "APP6", 0xFFE7: "APP7", 0xFFE8: "APP8",
    0xFFE9: "APP9", 0xFFEA: "APP10", 0xFFEB: "APP11",
    0xFFEC: "APP12", 0xFFED: "APP13", 0xFFEE: "APP14",
    0xFFEF: "APP15", 0xFFFE: "COM", 0xFFC2: "SOF2",
}

def parse_jpeg_segments_with_data(file_path):
    raw_segments = []
    with open(file_path, 'rb') as f:
        data = f.read()
        i = 0
        while i < len(data):
            if data[i] == 0xFF:
                while i + 1 < len(data) and data[i + 1] == 0xFF:
                    i += 1
                if i + 1 >= len(data):
                    break
                marker = (data[i] << 8) | data[i + 1]
                start = i
                i += 2
                if marker not in JPEG_MARKERS:
                    continue
                name = JPEG_MARKERS[marker]

                if marker in [0xFFD8, 0xFFD9]:
                    continue

                if i + 2 > len(data): break
                length = struct.unpack(">H", data[i:i + 2])[0]
                end = i + length
                if end > len(data):
                    end = len(data)
                segment_data = data[start:end]
                i = end

                raw_segments.append((name, segment_data))
            else:
                i += 1

    name_counter = defaultdict(int)
    total_counts = defaultdict(int)
    for name, _ in raw_segments:
        total_counts[name] += 1

    final_segments = []
    for name, segment_data in raw_segments:
        if total_counts[name] > 1:
            index = name_counter[name]
            name_counter[name] += 1
            display_name = f"{name}[{index}]"
        else:
            display_name = name
        final_segments.append((display_name, segment_data))
    return final_segments

def to_ascii(data):
    return ''.join([chr(b) if 32 <= b < 127 else '.' for b in data[:100]])

def hex_str(data):
    return ' '.join(f"{b:02X}" for b in data)

def hex_compact_str(data):
    return ''.join(f"{b:02X}" for b in data)

def scan_folder_for_jpegs(folder_path):
    print(f"📂 입력한 폴더 경로: {os.path.abspath(folder_path)}\n")

    all_segment_names = set()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.jfif')):
                full_path = os.path.join(root, file)
                segments = parse_jpeg_segments_with_data(full_path)

                for name, _ in segments:
                    if not name.startswith("SOS"):
                        all_segment_names.add(name)

    ordered_names = sorted(all_segment_names)
    header = ["Subfolder Name", "File Name", "segment_order_string"]
    for name in ordered_names:
        header.append(name)
    header.append("first_1024_bytes")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', 'jfif')):
                    subfolder = os.path.basename(root)
                    full_path = os.path.join(root, file)
                    segments = parse_jpeg_segments_with_data(full_path)

                    segment_order = []
                    segment_data_dict = {}

                    for name, data in segments:
                        segment_order.append(name)
                        if not name.startswith("SOS") and name not in segment_data_dict:
                            segment_data_dict[name] = hex_str(data)

                    with open(full_path, 'rb') as f:
                        raw_data = f.read()
                        first_1024 = hex_compact_str(raw_data[:1024])

                    row = [subfolder, file, ', '.join(segment_order)]
                    for name in ordered_names:
                        row.append(segment_data_dict.get(name, ""))
                    row.append(first_1024)

                    writer.writerow(row)

# 실행 설정
folder_path = "C:/Users/KIM/바탕 화면/본심/실험/jpeg/dataset"
csv_filename = "250517_1024최종데이터셋.csv"
scan_folder_for_jpegs(folder_path)
