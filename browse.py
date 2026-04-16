import os
import cv2
import numpy as np


def create_pillarbox(image_4x3, target_width=1920, target_height=1080):
    source_height, source_width, _channels = image_4x3.shape

    if source_height == target_height and source_width == target_width:
        return image_4x3

    # Scale
    # TODO: do better job of determining scaled width and height, and correctly fill scaled_height
    scaled_width = int(target_height / source_height * source_width)
    scaled_height = target_height
    resized_img = cv2.resize(
        image_4x3, (scaled_width, target_height), interpolation=cv2.INTER_AREA
    )

    # Convert to LAB for better color blending
    img_lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2Lab).astype(np.float64)

    # Center in canvas
    canvas_lab = np.zeros((target_height, target_width, 3), dtype=np.float64)
    pad_x = (target_width - scaled_width) // 2
    canvas_lab[:, pad_x : pad_x + scaled_width] = img_lab

    # Integral Image of LAB data
    integral = cv2.integral(img_lab)

    def fill_side(indices, is_left):
        for x in indices:
            # Normalized distance (0 to 1) across the pillarbox
            dist_px = (pad_x - 1 - x) if is_left else (x - (pad_x + scaled_width))
            norm_dist = dist_px / pad_x

            # Determine size of blur box
            r_horiz = int(norm_dist * scaled_width * 0.25)
            r_vert = int(1 + norm_dist ** 2 * scaled_height * 0.25)

            # Sampling boundaries
            edge_x = 0 if is_left else (scaled_width - 1)
            x1 = np.clip(edge_x - r_horiz, 0, scaled_width - 1)
            x2 = np.clip(edge_x + r_horiz, 0, scaled_width - 1)

            y_coords = np.arange(target_height)
            y1 = np.clip(y_coords - r_vert, 0, target_height - 1)
            y2 = np.clip(y_coords + r_vert, 0, target_height - 1)

            # Summed Area Table lookup
            A = integral[y1, x1]
            B = integral[y1, x2 + 1]
            C = integral[y2 + 1, x1]
            D = integral[y2 + 1, x2 + 1]

            counts = (y2 - y1 + 1)[:, None] * (x2 - x1 + 1)
            canvas_lab[:, x] = (D - B - C + A) / counts

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
