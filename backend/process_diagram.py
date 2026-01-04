#!/usr/bin/env python3
"""
Simple OpenCV-based prototype to convert a photographed (handwritten) diagram
into a cleaned vector SVG containing canonical shapes and lines.

This is a baseline; later we can replace parts with ML-based detectors.
"""
import io
import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import svgwrite
except Exception:
    svgwrite = None


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    arr = np.array(img)
    # Convert RGB->BGR for OpenCV
    return arr[:, :, ::-1].copy()


def preprocess_for_contours(cv_img: np.ndarray, max_dim=1200) -> Tuple[np.ndarray, float]:
    h, w = cv_img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold (foreground dark on white background) -> invert to get white on black
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 9)
    # Morphological close to join strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed, scale


def contours_to_svg_elements(bin_img: np.ndarray, min_area: int = 200) -> Tuple[list, Tuple[int, int]]:
    # Find contours
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bin_img.shape[:2]
    elems = []

    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon=0.02 * peri, closed=True)
        x, y, ww, hh = cv2.boundingRect(approx)

        # Try to detect common primitives
        if len(approx) == 4:
            # Rectangle-like
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            elems.append({"type": "rect_poly", "pts": pts, "bbox": (x, y, ww, hh), "area": area})
        elif len(approx) <= 8 and area > 800:
            # Possibly ellipse / rounded rect
            if cnt.shape[0] >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (MA, ma), angle = ellipse
                elems.append({"type": "ellipse", "cx": cx, "cy": cy, "rx": MA / 2.0, "ry": ma / 2.0, "angle": angle, "bbox": (x, y, ww, hh), "area": area})
            else:
                pts = cnt.reshape(-1, 2).tolist()
                elems.append({"type": "polyline", "pts": pts, "bbox": (x, y, ww, hh), "area": area})
        else:
            # Generic polyline
            pts = cnt.reshape(-1, 2).tolist()
            elems.append({"type": "polyline", "pts": pts, "bbox": (x, y, ww, hh), "area": area})

    # Sort by area descending so big shapes draw first
    elems.sort(key=lambda e: -e.get("area", 0))
    return elems, (w, h)


def generate_svg_string(elems: list, canvas_size: Tuple[int, int]) -> str:
    w, h = canvas_size
    if svgwrite is None:
        # Fallback: simple manual SVG string building
        parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
        parts.append("<g fill='none' stroke='black' stroke-width='2'>")
        for e in elems:
            if e["type"] == "rect_poly":
                pts = " ".join(f"{x},{y}" for x, y in e["pts"])
                parts.append(f"<polygon points='{pts}' />")
            elif e["type"] == "ellipse":
                parts.append(f"<ellipse cx='{e['cx']:.1f}' cy='{e['cy']:.1f}' rx='{e['rx']:.1f}' ry='{e['ry']:.1f}' transform='rotate({e['angle']:.1f} {e['cx']:.1f} {e['cy']:.1f})' />")
            elif e["type"] == "polyline":
                pts = " ".join(f"{int(x)},{int(y)}" for x, y in e["pts"])[:10000]
                parts.append(f"<polyline points='{pts}' stroke-linejoin='round' stroke-linecap='round' />")
        parts.append("</g>")
        parts.append("</svg>")
        return "\n".join(parts)

    dwg = svgwrite.Drawing(size=(w, h), profile='tiny')
    grp = dwg.add(dwg.g(fill='none', stroke='black', stroke_width=2, stroke_linejoin='round', stroke_linecap='round'))
    for e in elems:
        if e["type"] == "rect_poly":
            pts = [(x, y) for x, y in e["pts"]]
            grp.add(dwg.polygon(points=pts))
        elif e["type"] == "ellipse":
            grp.add(dwg.ellipse(center=(e['cx'], e['cy']), r=(e['rx'], e['ry']), transform=f'rotate({e['angle']} {e['cx']} {e['cy']})'))
        elif e["type"] == "polyline":
            pts = [(int(x), int(y)) for x, y in e["pts"]]
            # simplify polyline to avoid huge SVGs
            if len(pts) > 200:
                step = max(1, len(pts) // 200)
                pts = pts[::step]
            grp.add(dwg.polyline(points=pts))
    return dwg.tostring()


def image_stream_to_svg(file_stream) -> str:
    try:
        img = Image.open(file_stream)
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")

    cv_img = pil_to_cv2(img)
    bin_img, scale = preprocess_for_contours(cv_img)
    elems, canvas = contours_to_svg_elements(bin_img)

    # Note: elements are on scaled canvas; use canvas size directly
    svg = generate_svg_string(elems, canvas)
    return svg


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: process_diagram.py <input_image> <output_svg>")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    with open(infile, 'rb') as f:
        svg = image_stream_to_svg(f)

    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(svg)

    print(f"Saved SVG to {outfile}")
