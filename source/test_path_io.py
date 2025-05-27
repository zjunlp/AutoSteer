import os
import json
import argparse

def check_jsonl_readable(path, max_lines=3):
    assert os.path.isfile(path), f"[ERROR] File not found: {path}"
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] Line {i+1} in {path} is not valid JSON: {e}")
            if i + 1 >= max_lines:
                break
    print(f"[OK] JSONL file valid: {path}")

def check_dir_exists(path):
    assert os.path.isdir(path), f"[ERROR] Directory not found: {path}"
    print(f"[OK] Directory exists: {path}")

def check_dir_writable(path):
    os.makedirs(path, exist_ok=True)
    test_file = os.path.join(path, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"[OK] Directory writable: {path}")
    except Exception as e:
        raise PermissionError(f"[ERROR] Cannot write to directory {path}: {e}")

def check_log_path(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, 'w') as f:
            f.write("test log")
        os.remove(log_path)
        print(f"[OK] Log file path writable: {log_path}")
    except Exception as e:
        raise PermissionError(f"[ERROR] Cannot write log file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test dataset/model/log paths without training")

    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to main dataset jsonl file')
    parser.add_argument('--dataset_PIC_path', type=str, required=True,
                        help='Path to toxic-image dataset jsonl file')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Path to pretrained model cache')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output dir (test for writability)')


    args = parser.parse_args()

    check_jsonl_readable(args.dataset_path)
    check_jsonl_readable(args.dataset_PIC_path)
    check_dir_exists(args.cache_dir)
    check_dir_writable(args.output_dir)

if __name__ == "__main__":
    main()