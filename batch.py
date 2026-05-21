import os
from pathlib import Path
import cv2
import sys
from browse import create_pillarbox
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_image(file_path, target_width, target_height, root_path):
    """
    Worker function: Processes a single image file.
    Returns a status string and the relative path for clean console printing.
    """
    rel_path = file_path.relative_to(root_path)
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            return "SKIPPED_DECODE", rel_path, "Could not decode image data"

        h, w = img.shape[:2]
        if h == target_height and w == target_width:
            return "SKIPPED_MATCH", rel_path, "Already target size"

        # Run your pipeline (using the off-by-one fixed function!)
        processed_img = create_pillarbox(img, target_width, target_height)

        success = cv2.imwrite(str(file_path), processed_img)
        if success:
            return "SUCCESS", rel_path, f"({w}x{h} -> {target_width}x{target_height})"
        else:
            return "ERROR_WRITE", rel_path, "Failed to write file back to disk"

    except Exception as e:
        return "FAILED", rel_path, str(e)


def batch_process_directory(directory_path, target_width=1920, target_height=1080):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    path = Path(directory_path)

    if not path.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    print(f"Scanning '{path.resolve()}' for images...")

    # 1. Gather all target files first
    tasks = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in valid_extensions]
    total_images = len(tasks)

    print(f"Found {total_images} images. Starting parallel processing thread pool...\n")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 2. Maximize performance by using an Executor Pool
    # By default, max_workers matches your total CPU core count
    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the queue
        futures = {
            executor.submit(process_single_image, file_path, target_width, target_height, path): file_path
            for file_path in tasks
        }

        # 3. As each process finishes, print its result in real-time
        for future in as_completed(futures):
            status, rel_file, message = future.result()

            if status == "SUCCESS":
                print(f"[DONE] {rel_file} -> {message}")
                processed_count += 1
            elif "SKIPPED" in status:
                print(f"[SKIPPED] {rel_file} -> {message}")
                if status == "SKIPPED_MATCH":
                    skipped_count += 1
                else:
                    error_count += 1
            else:
                print(f"[FAILED] {rel_file} -> Error: {message}")
                error_count += 1

    # Print summary
    print("\n" + "="*40)
    print("Parallel Processing Completed Summary:")
    print(f"  Successfully Processed: {processed_count}")
    print(f"  Skipped (Already Match): {skipped_count}")
    print(f"  Errors / Failures:      {error_count}")
    print("="*40)


if __name__ == "__main__":
    target_folder = sys.argv[1]
    batch_process_directory(target_folder, target_width=1920, target_height=1080)