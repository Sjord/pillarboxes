import os
import cv2
import numpy as np
from PIL import Image
from skimage import color  # Import skimage.color for CIELAB conversion


def calculate_average_color(image_path):
    """
    Calculates the average color of an image in CIELAB color space
    and returns it as a BGR tuple suitable for OpenCV. This implementation
    uses the skimage library for color space conversions, following
    approaches seen in Stack Overflow answers for CIELAB conversion.
    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: A tuple (B, G, R) representing the average color.
    """
    try:
        with Image.open(image_path) as img_pil:
            # Convert to RGB if not already.
            img_pil = img_pil.convert("RGB")
            # Convert pixel values to float between 0 and 1, as skimage.color.rgb2lab expects this.
            img_np_rgb = np.array(img_pil) / 255.0

            # Convert the RGB image data to CIELAB color space.
            img_lab = color.rgb2lab(img_np_rgb)

            # Calculate the average of each channel (L, a, b) in CIELAB
            avg_lab = np.mean(img_lab, axis=(0, 1))

            # Convert the average CIELAB color back to sRGB.
            avg_srgb = color.lab2rgb(
                np.expand_dims(np.expand_dims(avg_lab, axis=0), axis=0)
            )[0, 0]

            # Convert sRGB values (0-1 range) back to 0-255 integer range for display.
            avg_color_rgb = (np.clip(avg_srgb * 255, 0, 255)).astype(int)

            # OpenCV uses BGR format, so reorder the RGB tuple
            avg_color_bgr = (
                int(avg_color_rgb[2]),
                int(avg_color_rgb[1]),
                int(avg_color_rgb[0]),
            )
            return avg_color_bgr
    except Exception as e:
        print(f"Error calculating average color for {image_path}: {e}")
        return (0, 0, 0)  # Return black in case of error


def crop_to_4x3(image):
    """
    Crops an image to a 4x3 aspect ratio, centering the crop.
    Args:
        image (numpy.ndarray): The input image (OpenCV format).
    Returns:
        numpy.ndarray: The cropped image.
    """
    h, w, _ = image.shape
    target_aspect_ratio = 4 / 3

    current_aspect_ratio = w / h

    if current_aspect_ratio > target_aspect_ratio:
        # Image is wider than 4x3, crop horizontally
        new_width = int(h * target_aspect_ratio)
        start_x = (w - new_width) // 2
        cropped_image = image[:, start_x : start_x + new_width]
    elif current_aspect_ratio < target_aspect_ratio:
        # Image is taller than 4x3, crop vertically
        new_height = int(w / target_aspect_ratio)
        start_y = (h - new_height) // 2
        cropped_image = image[start_y : start_y + new_height, :]
    else:
        # Image is already 4x3
        cropped_image = image

    return cropped_image


def create_pillarboxed_image(
    cropped_4x3_image, background_color, target_width=1280, target_height=720
):
    """
    Creates a 16x9 image with the 4x3 image centered and background color filling the sides.
    Args:
        cropped_4x3_image (numpy.ndarray): The 4x3 cropped image.
        background_color (tuple): The (B, G, R) average color for the background.
        target_width (int): Desired width of the output 16x9 window.
        target_height (int): Desired height of the output 16x9 window.
    Returns:
        numpy.ndarray: The 16x9 image with the 4x3 image and pillarbox background.
    """
    # Create a blank 16x9 canvas with the average background color
    pillarboxed_image = np.full(
        (target_height, target_width, 3), background_color, dtype=np.uint8
    )

    # Calculate dimensions for the 4x3 image within the 16x9 frame
    # The 4x3 image will be scaled to fit the height of the 16x9 frame
    scaled_4x3_height = target_height
    scaled_4x3_width = int(scaled_4x3_height * (4 / 3))

    # Resize the 4x3 image to fit the calculated dimensions
    resized_4x3_image = cv2.resize(
        cropped_4x3_image,
        (scaled_4x3_width, scaled_4x3_height),
        interpolation=cv2.INTER_AREA,
    )

    # Calculate padding for centering the 4x3 image
    padding_x = (target_width - scaled_4x3_width) // 2

    # Paste the resized 4x3 image onto the center of the pillarboxed canvas
    pillarboxed_image[0:scaled_4x3_height, padding_x : padding_x + scaled_4x3_width] = (
        resized_4x3_image
    )

    return pillarboxed_image


def list_image_files():
    corpus_dir = "./corpus"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # Get a list of all image files in the corpus directory
    image_files = []
    if not os.path.exists(corpus_dir):
        raise Exception(f"Error: The directory '{corpus_dir}' does not exist.")

    for filename in os.listdir(corpus_dir):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(corpus_dir, filename))

    if not image_files:
        raise Exception(
            f"No image files found in '{corpus_dir}'. Please add some images."
        )

    print(f"Found {len(image_files)} images in '{corpus_dir}'.")
    return image_files


def main():
    image_files = list_image_files()

    print(
        "Press any key or click the mouse to view the next photo. Close the window or press 'q' to exit."
    )

    cv2.namedWindow("Photo Viewer", cv2.WINDOW_NORMAL)  # Allow window to be resizable

    for i, image_path in enumerate(image_files):
        print(
            f"Displaying: {os.path.basename(image_path)} ({i + 1}/{len(image_files)})"
        )

        avg_color_bgr = calculate_average_color(image_path)

        # Read the image using OpenCV (for subsequent processing)
        original_image_cv = cv2.imread(image_path)
        if original_image_cv is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Crop the image to 4x3
        cropped_4x3_image = crop_to_4x3(original_image_cv)

        # Create the 16x9 pillarboxed image
        # Using a fixed target size for the window, e.g., 1280x720 (720p 16:9)
        display_image = create_pillarboxed_image(
            cropped_4x3_image, avg_color_bgr, 1280, 720
        )

        # Show the image
        cv2.imshow("Photo Viewer", display_image)

        # Wait for a key press (0 means wait indefinitely) or mouse click
        # cv2.waitKey() returns the ASCII value of the pressed key
        # For mouse events, the window must be active. Any key press will advance.
        key = cv2.waitKey(0) & 0xFF  # Use & 0xFF for cross-platform compatibility

        # Optional: You could add a specific key to exit, e.g., 'q'
        if key == ord("q"):
            print("Exiting viewer.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
