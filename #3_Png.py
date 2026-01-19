import os
import struct
import csv

# PNG 공식 및 비공식 청크
PNG_CHUNKS = [
    "IHDR", "PLTE", "IDAT", "IEND", "tRNS", "gAMA", "cHRM", "sRGB", "sBIT", "bKGD",
    "pHYs", "hIST", "tIME", "tEXt", "zTXt", "iTXt", "eXIf", "acTL", "fcTL", "fdAT",
    "vpAg", "caNv", "mkBF"
]

def parse_png_chunks(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"Not a valid PNG file: {file_path}")

    chunks = []
    i = 8
    pre_idat_data = bytearray()
    found_idat = False

    while i < len(data):
        if i + 8 > len(data):
            break

        length = struct.unpack(">I", data[i:i+4])[0]
        chunk_type = data[i+4:i+8].decode('latin1')
        chunk_data = data[i+8:i+8+length]
        chunks.append((chunk_type, chunk_data))

        if not found_idat:
            if chunk_type != "IDAT":
                pre_idat_data.extend(data[i:i+8+length+4])
            else:
                found_idat = True

        i += 8 + length + 4

    return chunks, pre_idat_data

def hex_str(data, limit=100):
    return ' '.join(f"{b:02X}" for b in data[:limit])

def scan_folder_for_pngs(folder_path, csv_filename):
    print(f"📂 입력한 폴더 경로: {os.path.abspath(folder_path)}\n")

    all_chunk_names = set()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                try:
                    chunks, _ = parse_png_chunks(full_path)
                    for chunk_type, _ in chunks:
                        all_chunk_names.add(chunk_type)
                except Exception as e:
                    print(f"⚠️ {file}: {e}")

    ordered_chunk_names = sorted(all_chunk_names)

    # ✅ IDAT, IEND 제외하고 hex 컬럼 구성
    header = ["Subfolder Name", "File Name", "Chunk Order", "Pre-IDAT Hex"]
    for name in ordered_chunk_names:
        if name not in ("IDAT", "IEND"):
            header.append(f"{name}_hex")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.png'):
                    subfolder = os.path.basename(root)
                    full_path = os.path.join(root, file)
                    try:
                        chunks, pre_idat_data = parse_png_chunks(full_path)

                        chunk_order = []
                        chunk_hex_dict = {}

                        for chunk_type, chunk_data in chunks:
                            chunk_order.append(chunk_type)
                            if chunk_type not in ("IDAT", "IEND") and chunk_type not in chunk_hex_dict:
                                chunk_hex_dict[chunk_type] = hex_str(chunk_data)

                        row = [subfolder, file, ', '.join(chunk_order), hex_str(pre_idat_data)]
                        for name in ordered_chunk_names:
                            if name not in ("IDAT", "IEND"):
                                row.append(chunk_hex_dict.get(name, ""))
                        writer.writerow(row)

                    except Exception as e:
                        print(f"⚠️ {file}: {e}")

# 실행 설정
folder_path = "C:/Users/KIM/바탕 화면/본심/실험/png/dataset"
csv_filename = "png뭐지.csv"
scan_folder_for_pngs(folder_path, csv_filename)
