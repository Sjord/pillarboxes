import os
import cv2
import numpy as np


def resize(image, target_width=1920, target_height=1080):
    source_height, source_width = image.shape[:2]

    if source_height == target_height and source_width == target_width:
        return image

    # 1. Calculate the scaling ratio for both dimensions
    # We want the ratio that fits the image entirely within the target box
    ratio_w = target_width / source_width
    ratio_h = target_height / source_height
    scale = min(ratio_w, ratio_h)

    # 2. Determine new dimensions based on the aspect ratio scale
    scaled_width = int(source_width * scale)
    scaled_height = int(source_height * scale)

    # 3. Resize the image
    # Use INTER_AREA for shrinking (better quality) and INTER_CUBIC for enlarging
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized_img = cv2.resize(image, (scaled_width, scaled_height), interpolation=interp)
    return resized_img


def create_pillarbox(image_4x3, target_width=1920, target_height=1080):
    source_height, source_width, _channels = image_4x3.shape

    if source_height == target_height and source_width == target_width:
        return image_4x3

    # Scale
    resized_img = resize(image_4x3, target_width=1920, target_height=1080)
    scaled_height, scaled_width = resized_img.shape[:2]

    # Convert to LAB for better color blending
    img_lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2Lab).astype(np.float64)

    # Center in canvas
    canvas_lab = np.zeros((target_height, target_width, 3), dtype=np.float64)
    pad_x = (target_width - scaled_width) // 2
    canvas_lab[:, pad_x : pad_x + scaled_width] = img_lab

    # Integral Image of LAB data
    integral = cv2.integral(img_lab)

    def fill_side(indices, is_left):
        def get_box_avg(rh, rv):
            """Helper to perform the vectorized SAT lookup for a given radius pair."""
            # Sampling boundaries
            edge_x = 0 if is_left else (scaled_width - 1)
            x1 = np.clip(edge_x - rh, 0, scaled_width - 1)
            x2 = np.clip(edge_x + rh, 0, scaled_width - 1)

            y_coords = np.arange(target_height)
            y1 = np.clip(y_coords - rv, 0, target_height - 1)
            y2 = np.clip(y_coords + rv, 0, target_height - 1)

            # SAT lookup (A, B, C, D corners)
            A = integral[y1, x1]
            B = integral[y1, x2 + 1]
            C = integral[y2 + 1, x1]
            D = integral[y2 + 1, x2 + 1]

            # Area calculation
            area = (y2 - y1 + 1)[:, None] * (x2 - x1 + 1)
            return (D - B - C + A) / area

        for x in indices:
            dist_px = (pad_x - 1 - x) if is_left else (x - (pad_x + scaled_width))
            norm_dist = dist_px / pad_x

            # Base radii
            base_rh = norm_dist * scaled_width * 0.25
            base_rv = 1 + (norm_dist ** 2) * scaled_height * 0.25

            # Calculate 3 boxes with varying sizes (0.6x, 1.0x, 1.4x)
            # This spread helps simulate a bell-curve weight distribution
            avg1 = get_box_avg(int(base_rh * 0.6), int(base_rv * 0.6))
            avg2 = get_box_avg(int(base_rh), int(base_rv))
            avg3 = get_box_avg(int(base_rh * 1.4), int(base_rv * 1.4))

            # Blend the three boxes
            canvas_lab[:, x] = (avg1 + avg2 + avg3) / 3.0

    fill_side(np.arange(pad_x), True)
    fill_side(np.arange(pad_x + scaled_width, target_width), False)

    # Convert back to BGR
    result_bgr = cv2.cvtColor(canvas_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result_bgr


def main():
    corpus_dir = "./corpus"
    if not os.path.exists(corpus_dir):
        print(f"Directory {corpus_dir} not found.")
        return

    files = [
        os.path.join(corpus_dir, f)
        for f in os.listdir(corpus_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_path in files:
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = create_pillarbox(img, 1920, 1080)

        cv2.imshow("Dynamic Pillarbox", result)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
