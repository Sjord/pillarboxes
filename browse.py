import os
import cv2
import numpy as np


def create_pillarbox(image_4x3, target_width=1920, target_height=1080):
    h_target, w_target = target_height, target_width

    # 1. Scale and Center
    scaled_w = int(h_target * (4 / 3))
    resized_img = cv2.resize(
        image_4x3, (scaled_w, h_target), interpolation=cv2.INTER_AREA
    )

    # 2. Convert to LAB for better color blending
    # (LAB prevents "muddy" transitions in the blur)
    img_lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2Lab).astype(np.float64)

    canvas_lab = np.zeros((h_target, w_target, 3), dtype=np.float64)
    pad_x = (w_target - scaled_w) // 2
    canvas_lab[:, pad_x : pad_x + scaled_w] = img_lab

    # 3. Integral Image of LAB data
    integral = cv2.integral(img_lab)

    def fill_side(indices, is_left):
        for x in indices:
            # Normalized distance (0 to 1) across the pillarbox
            dist_px = (pad_x - 1 - x) if is_left else (x - (pad_x + scaled_w))
            norm_dist = dist_px / pad_x

            # QUADRATIC RADIUS: Radius stays small near image, grows fast far away
            # We use an asymmetric box: Wide horizontally, short vertically
            base_r = (norm_dist**2) * (scaled_w * 0.5)
            r_horiz = int((norm_dist) * (scaled_w * 0.5))
            r_vert = int(1 + base_r * 0.25)  # 1:4 aspect ratio to preserve horizontal lines

            edge_x = 0 if is_left else (scaled_w - 1)

            # Sampling boundaries
            x1 = np.clip(edge_x - r_horiz, 0, scaled_w - 1)
            x2 = np.clip(edge_x + r_horiz, 0, scaled_w - 1)

            y_coords = np.arange(h_target)
            y1 = np.clip(y_coords - r_vert, 0, h_target - 1)
            y2 = np.clip(y_coords + r_vert, 0, h_target - 1)

            # Summed Area Table lookup
            A = integral[y1, x1]
            B = integral[y1, x2 + 1]
            C = integral[y2 + 1, x1]
            D = integral[y2 + 1, x2 + 1]

            counts = (y2 - y1 + 1)[:, None] * (x2 - x1 + 1)
            canvas_lab[:, x] = (D - B - C + A) / counts

    fill_side(np.arange(pad_x), True)
    fill_side(np.arange(pad_x + scaled_w, w_target), False)

    # 4. Convert back to BGR
    result_bgr = cv2.cvtColor(canvas_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result_bgr


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
        result = create_pillarbox(cropped, 1920, 1080)

        cv2.imshow("Dynamic Pillarbox", result)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
