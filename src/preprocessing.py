import cv2

def load_image(image_path):
    """
    Reads an image from a given file path.
    Input: File path (string).
    Output: NumPy array (BGR image).
    Raises ValueError if the image cannot be found or read.
    """

    img= cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Can not read image at path:{image_path}")
    return img

def convert_to_gray(img_bgr):
    """
    Converts a BGR image to Grayscale.
    Input: BGR image matrix (3 channels).
    Output: Grayscale image matrix (1 channel, 2 dimensions).
    """
    return cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

def convert_to_rgb(image_bgr):
    """
    Converts a BGR image to RGB (Useful for Matplotlib display).
    Input: BGR image matrix.
    Output: RGB image matrix.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def convert_to_hsv(image_bgr):
    """
    Converts a BGR image to HSV (Useful for color and brightness separation).
    Input: BGR image matrix.
    Output: HSV image matrix.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)