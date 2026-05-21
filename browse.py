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


def create_pillarbox(image, target_width=1920, target_height=1080):
    source_height, source_width = image.shape[:2]

    if source_height == target_height and source_width == target_width:
        return image

    # 1. Determine if we need Letterbox (top/bottom) instead of Pillarbox (left/right)
    # We compare aspect ratios to see which direction needs padding
    needs_transpose = (source_width / source_height) > (target_width / target_height)

    if needs_transpose:
        # Swap target dimensions and transpose input so we always work on vertical pillarboxes
        image = cv2.transpose(image)
        target_width, target_height = target_height, target_width

    # 2. Scale image to fit the target height perfectly
    resized_img = resize(image, target_width=target_width, target_height=target_height)

    # Force the resized image height to match target_height exactly in case of off-by-one rounding
    if resized_img.shape[0] != target_height:
        resized_img = cv2.resize(resized_img, (resized_img.shape[1], target_height), interpolation=cv2.INTER_LINEAR)

    scaled_height, scaled_width = resized_img.shape[:2]

    # Convert to LAB for better color blending
    img_lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2Lab).astype(np.float64)

    # Center in canvas
    canvas_lab = np.zeros((target_height, target_width, 3), dtype=np.float64)
    pad_x = (target_width - scaled_width) // 2
    canvas_lab[:, pad_x : pad_x + scaled_width] = img_lab

    # 3. Define a unified Left-Side Blur Engine
    def blur_left_side(canvas, source_img):
        integral = cv2.integral(source_img)

        # We only ever loop from 0 to pad_x
        for x in range(pad_x):
            dist_px = pad_x - 1 - x
            norm_dist = dist_px / pad_x

            base_rh = norm_dist * scaled_width * 0.25
            rv_linear = 3 + norm_dist * scaled_height * 0.25
            rv_quad = 3 + (norm_dist ** 2) * scaled_height * 0.25

            # Sampling bounds (always anchored to edge_x = 0)
            x1 = 0
            x2 = np.clip(int(base_rh * 1.4), 0, scaled_width - 1)

            y_coords = np.arange(target_height)

            # Helper to quickly query the SAT for a specific vertical radius
            def get_blur_v(rv):
                y1 = np.clip(y_coords - int(rv), 0, target_height - 1)
                y2 = np.clip(y_coords + int(rv), 0, target_height - 1)

                A = integral[y1, x1]
                B = integral[y1, x2 + 1]
                C = integral[y2 + 1, x1]
                D = integral[y2 + 1, x2 + 1]

                area = (y2 - y1 + 1)[:, None] * (x2 - x1 + 1)
                return (D - B - C + A) / area

            # Average our 4 blur steps cleanly
            sums = (get_blur_v(rv_linear) +
                    get_blur_v(rv_linear * 1.4) +
                    get_blur_v(rv_quad) +
                    get_blur_v(rv_quad * 1.4))

            canvas[:, x] = sums / 4.0

    # 4. Execute the blurs using flips
    # Blur the actual left side
    blur_left_side(canvas_lab, img_lab)

    # Flip canvas and source horizontally, blur the "new" left side (which is the right side), flip back
    canvas_lab = cv2.flip(canvas_lab, 1)
    img_lab_flipped = cv2.flip(img_lab, 1)
    blur_left_side(canvas_lab, img_lab_flipped)
    canvas_lab = cv2.flip(canvas_lab, 1)

    # 5. Convert back to BGR
    result_bgr = cv2.cvtColor(canvas_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # If we transposed at the beginning, transpose back to restore original orientation
    if needs_transpose:
        result_bgr = cv2.transpose(result_bgr)

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
