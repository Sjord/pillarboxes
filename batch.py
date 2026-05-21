import os
from pathlib import Path
import cv2
import sys
from browse import create_pillarbox


def batch_process_directory(directory_path, target_width=1920, target_height=1080):
    """
    Recursively finds all images in a directory, processes them with 
    create_pillarbox, and overwrites the original files.
    """
    # Supported image extensions (case-insensitive)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    path = Path(directory_path)
    if not path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    print(f"Scanning '{path.resolve()}' for images...")
    
    # Track metrics for the summary
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # rglob("*") recursively finds all files and folders
    for file_path in path.rglob("*"):
        # Check if it's a file and has an image extension
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            print(f"Processing: {file_path.relative_to(path)}", end="", flush=True)
            
            try:
                # Read the image
                # Note: Using str(file_path) for compatibility with OpenCV path handling
                img = cv2.imread(str(file_path))
                
                if img is None:
                    print(" -> [SKIPPED] (Could not decode image data)")
                    error_count += 1
                    continue
                
                # Check dimensions to see if it even needs processing
                h, w = img.shape[:2]
                if h == target_height and w == target_width:
                    print(" -> [SKIPPED] (Already target size)")
                    skipped_count += 1
                    continue

                # Run your blur padding pipeline
                processed_img = create_pillarbox(img, target_width, target_height)
                
                # Overwrite the original file
                success = cv2.imwrite(str(file_path), processed_img)
                
                if success:
                    print(f" -> [DONE] ({w}x{h} -> {target_width}x{target_height})")
                    processed_count += 1
                else:
                    print(" -> [ERROR] (Failed to write file back to disk)")
                    error_count += 1
                    
            except Exception as e:
                print(f" -> [FAILED] (Unexpected error: {str(e)})")
                error_count += 1

    # Print out a little execution summary
    print("\n" + "="*40)
    print("Processing Completed Summary:")
    print(f"  Successfully Processed: {processed_count}")
    print(f"  Skipped (Already Match): {skipped_count}")
    print(f"  Errors / Failures:      {error_count}")
    print("="*40)


if __name__ == "__main__":
    target_folder = sys.argv[1]
    batch_process_directory(target_folder, target_width=1920, target_height=1080)