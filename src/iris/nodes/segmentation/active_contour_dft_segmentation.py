from __future__ import annotations

import cv2
import numpy as np
import logging
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.fftpack import fft, ifft
from typing import Any, List, Tuple

from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.algorithm import Algorithm

logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename="segmentation.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ActiveContourDFTSegmentation(Algorithm):
    """Segmentation using Hough Transform, Active Contour, and DFT smoothing."""

    class Parameters(Algorithm.Parameters):
        """Parameter class for ActiveContourDFTSegmentation."""
        pass

    __parameters_type__ = Parameters

    CLASSES_MAPPING = {
        0: "eyeball",
        1: "iris",
        2: "pupil",
        3: "eyelashes",
    }

    def __init__(self, alpha: float = 0.01, beta: float = 10, callbacks: List[Any] = []) -> None:
        super().__init__(callbacks=callbacks)
        self.alpha = alpha
        self.beta = beta

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before segmentation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_initial_boundary(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Detects an initial circular iris boundary using Hough Transform."""
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=20, maxRadius=80
        )
        if circles is not None and len(circles[0]) > 0:
            circles = sorted(circles[0], key=lambda c: c[2], reverse=True)
            x, y, r = np.round(circles[0]).astype(int)
            return x, y, r
        logging.warning("No circles detected, using default estimate.")
        return image.shape[1] // 2, image.shape[0] // 2, min(image.shape) // 4

    def refine_boundary_active_contour(self, image: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
        """Refines the detected boundary using the Active Contour Model."""
        s = np.linspace(0, 2 * np.pi, 400)
        init_x = x + r * np.cos(s)
        init_y = y + r * np.sin(s)
        snake = active_contour(gaussian(image, 1), np.array([init_y, init_x]).T, alpha=self.alpha, beta=self.beta)
        return snake

    def smooth_contour_dft(self, contour: np.ndarray, keep_fraction: float = 0.2) -> np.ndarray:
        """Smooth the contour using Discrete Fourier Transform (DFT)."""
        transformed = fft(contour, axis=0)
        n = int(len(transformed) * keep_fraction)
        transformed[n:-n] = 0
        smoothed = np.real(ifft(transformed, axis=0))
        return smoothed

    def generate_segmentation_map(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Generates a binary segmentation map from the refined contour."""
        mask = np.zeros_like(image, dtype=np.uint8)
        rr, cc = circle_perimeter(int(contour[:, 0].mean()), int(contour[:, 1].mean()), int(np.ptp(contour[:, 0]) / 2))
        rr, cc = np.clip(rr, 0, image.shape[0] - 1), np.clip(cc, 0, image.shape[1] - 1)  # Ensure bounds
        mask[rr, cc] = 1
        return mask

    def run(self, image: IRImage) -> SegmentationMap:
        """Execute the segmentation pipeline on the given image."""
        logging.info("Running segmentation pipeline...")
        preprocessed_image = self.preprocess(image.img_data)
        logging.info("Preprocessing complete")
        x, y, r = self.detect_initial_boundary(preprocessed_image)
        logging.info(f"Initial boundary detected at ({x}, {y}) with radius {r}")
        refined_contour = self.refine_boundary_active_contour(preprocessed_image, x, y, r)
        logging.info("Active contour refinement complete")
        smoothed_contour = self.smooth_contour_dft(refined_contour)
        logging.info("DFT smoothing complete")
        segmentation_map = self.generate_segmentation_map(preprocessed_image, smoothed_contour)
        logging.info("Segmentation map generated successfully")
        return SegmentationMap(predictions=segmentation_map, index2class=self.CLASSES_MAPPING)