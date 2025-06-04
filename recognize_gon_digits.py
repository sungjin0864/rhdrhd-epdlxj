import cv2
import numpy as np
import os


def generate_templates(size: int = 32, font: int = cv2.FONT_HERSHEY_SIMPLEX):
    """Generate template images for digits and arithmetic operators."""
    symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"]
    templates = {}
    for sym in symbols:
        img = np.ones((size, size), dtype=np.uint8) * 255
        cv2.putText(img, sym, (4, size - 4), font, 1, 0, 2, cv2.LINE_AA)
        _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img_bin = cv2.bitwise_not(img_bin)
        templates[sym] = img_bin
    return templates


def preprocess(img: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (size, size))
    _, img_bin = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin)
    return img_bin


def recognize_symbol(image_path: str, templates: dict) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found")
    processed = preprocess(img)
    best_label = None
    best_score = float("inf")
    for label, templ in templates.items():
        diff = cv2.absdiff(processed, templ)
        score = diff.sum()
        if score < best_score:
            best_score = score
            best_label = label
    return best_label


def main(folder: str):
    templates = generate_templates()
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            label = recognize_symbol(path, templates)
            print(f"{file}: {label}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python recognize_gon_digits.py <image_folder>")
    else:
        main(sys.argv[1])
