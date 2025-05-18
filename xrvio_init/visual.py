import cv2
import numpy as np
from typing import List, Tuple, Dict

from xrvio_init.inertial import rodrigues_matrix


def extract_and_match(frames: List[np.ndarray],
                      max_features: int = 2000,
                      ratio: float = 0.75
                      ) -> Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]]:
    """
    Extract ORB features and match between consecutive frames using Lowe's ratio test.

    Args:
        frames: list of images (H,W,3) in uint8
        max_features: max number of ORB keypoints per image
        ratio: ratio test threshold

    Returns:
        matches: dict mapping (i,i+1) -> (pts1, pts2) where
                 pts1, pts2 are (M,2) arrays of matched keypoint coordinates
    """
    # Initialize ORB and BFMatcher
    orb = cv2.ORB_create(max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Detect and describe
    keypoints = []
    descriptors = []
    for img in frames:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    matches = {}
    for i in range(len(frames) - 1):
        des1, des2 = descriptors[i], descriptors[i+1]
        if des1 is None or des2 is None:
            continue
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good1, good2 = [], []
        for m, n in raw_matches:
            if m.distance < ratio * n.distance:
                pt1 = keypoints[i][m.queryIdx].pt
                pt2 = keypoints[i+1][m.trainIdx].pt
                good1.append(pt1)
                good2.append(pt2)
        if len(good1) >= 8:
            matches[(i, i+1)] = (np.array(good1, dtype=np.float64),
                                 np.array(good2, dtype=np.float64))
    return matches


def estimate_rel_rot(R_prior: np.ndarray,
                     pts1: np.ndarray,
                     pts2: np.ndarray,
                     K: np.ndarray
                     ) -> np.ndarray:
    """
    Refine a prior rotation by minimizing reprojection error (rotation-only).

    Args:
        R_prior: 3x3 prior rotation from frame i to j
        pts1, pts2: (M,2) matched image points
        K: (3x3) camera intrinsics

    Returns:
        R_refined: refined 3x3 rotation
    """
    from scipy.optimize import least_squares

    # Precompute normalized bearing vectors
    Kinv = np.linalg.inv(K)
    ones = np.ones((pts1.shape[0], 1))
    b1 = (Kinv @ np.hstack([pts1, ones]).T).T
    b2 = (Kinv @ np.hstack([pts2, ones]).T).T

    def residual(rvec):
        # small-angle update
        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            R_update = np.eye(3)
        else:
            axis = rvec / theta
            R_update = rodrigues_matrix(axis, theta)
        R = R_update @ R_prior
        # project b1 through R
        b1_rot = (R @ b1.T).T
        # residuals between bearing directions
        err = b2 - b1_rot
        return err.ravel()

    # initial zero update
    r0 = np.zeros(3)
    sol = least_squares(residual, r0, method='lm')
    theta = np.linalg.norm(sol.x)
    if theta < 1e-12:
        R_update = np.eye(3)
    else:
        R_update = rodrigues_matrix(sol.x / theta, theta)
    return R_update @ R_prior


def solve_translation(R: np.ndarray,
                      pts1: np.ndarray,
                      pts2: np.ndarray,
                      K: np.ndarray
                      ) -> np.ndarray:
    """
    Solve for translation (up to scale) given known rotation and feature matches.

    Args:
        R: 3x3 rotation from frame i to j
        pts1, pts2: (M,2) matched image points
        K: 3x3 intrinsics

    Returns:
        t: (3,) unit-norm translation vector
    """
    Kinv = np.linalg.inv(K)
    ones = np.ones((pts1.shape[0], 1))
    b1 = (Kinv @ np.hstack([pts1, ones]).T).T
    b2 = (Kinv @ np.hstack([pts2, ones]).T).T

    # Build linear system A t = 0, where each row is (b2 x (R b1))^T
    A = []
    for x1, x2 in zip(b1, b2):
        A.append(np.cross(x2, R @ x1))
    A = np.vstack(A)
    # Solve for nullspace vector
    _, _, Vt = np.linalg.svd(A)
    t = Vt[-1]
    return t / np.linalg.norm(t)
