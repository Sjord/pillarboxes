import os
import cv2
import numpy as np


def get_box_average(integral_img, x1, y1, x2, y2):
    """
    Uses the Integral Image to calculate the average color of a rectangle
    defined by (x1, y1) to (x2, y2) inclusive.
    """
    # Ensure coordinates are within image bounds
    h, w = integral_img.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, w - 2)
    y1, y2 = np.clip([y1, y2], 0, h - 2)

    # Standard Summed-Area Table formula: I(D) - I(B) - I(C) + I(A)
    # integral_img has shape (H+1, W+1)
    sum_val = (
        integral_img[y2 + 1, x2 + 1]
        - integral_img[y1, x2 + 1]
        - integral_img[y2 + 1, x1]
        + integral_img[y1, x1]
    )

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return sum_val / area


def create_dynamic_blur_pillarbox(image_4x3, target_width=1920, target_height=1080):
    # 1. Scale image_4x3 to fit the height of the target
    h_target, w_target = target_height, target_width
    h_src, w_src = image_4x3.shape[:2]

    scaled_w = int(h_target * (4 / 3))
    resized_img = cv2.resize(
        image_4x3, (scaled_w, h_target), interpolation=cv2.INTER_AREA
    )

    # 2. Setup Canvas
    canvas = np.zeros((h_target, w_target, 3), dtype=np.uint8)
    pad_x = (w_target - scaled_w) // 2
    canvas[:, pad_x : pad_x + scaled_w] = resized_img

    # 3. Prepare Integral Image (Summed-Area Table)
    # We use float64 to avoid overflow during summation
    integral = cv2.integral(resized_img.astype(np.float64))

    # 4. Process Pillarboxes (Left and Right)
    # x_coords represents the x-indices of the pillarbox pixels
    left_indices = np.arange(pad_x)
    right_indices = np.arange(pad_x + scaled_w, w_target)

    def fill_pillarbox(indices, is_left):
        for x in indices:
            # Distance to the image edge
            dist = (pad_x - 1 - x) if is_left else (x - (pad_x + scaled_w))
            radius = 2 * dist

            # The edge x-coordinate in the resized_img
            edge_x = 0 if is_left else (scaled_w - 1)

            # Define the box to average in the source image
            # The box is centered vertically at each y, and at the horizontal edge
            x1 = int(edge_x - radius)
            x2 = int(edge_x + radius)

            # For efficiency, we can calculate the column average once per x
            # Since the radius is the same for all y at this x
            y_coords = np.arange(h_target)
            y1 = (y_coords - radius).astype(int)
            y2 = (y_coords + radius).astype(int)

            # Clip y boundaries
            y1 = np.clip(y1, 0, h_target - 1)
            y2 = np.clip(y2, 0, h_target - 1)
            x1_c = np.clip(x1, 0, scaled_w - 1)
            x2_c = np.clip(x2, 0, scaled_w - 1)

            # Calculate box sums using the integral image
            # Formula: I(y2+1, x2+1) - I(y1, x2+1) - I(y2+1, x1) + I(y1, x1)
            A = integral[y1, x1_c]
            B = integral[y1, x2_c + 1]
            C = integral[y2 + 1, x1_c]
            D = integral[y2 + 1, x2_c + 1]

            sums = D - B - C + A
            counts = (y2 - y1 + 1)[:, None] * (x2_c - x1_c + 1)
            avg_colors = sums / counts

            canvas[:, x] = avg_colors.astype(np.uint8)

    fill_pillarbox(left_indices, True)
    fill_pillarbox(right_indices, False)

    return canvas


def crop_to_4x3(image):
    h, w = image.shape[:2]
    target_ratio = 4 / 3
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return image[:, offset : offset + new_w]
    else:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return image[offset : offset + new_h, :]


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

        cropped = crop_to_4x3(img)
        # Create 1920x1080 output with dynamic blur
        result = create_dynamic_blur_pillarbox(cropped, 1920, 1080)

        cv2.imshow("Dynamic Pillarbox", result)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
