"""Трекинг плоского объекта на видео по ключевым точкам.

Эталон объекта - первый кадр видео. Алгоритм на каждом кадре:
  1. Если объект был найден на прошлом кадре, его точки переносятся на новый кадр
     оптическим потоком Лукаса-Канаде (быстро, устойчиво к размытию и малому размеру).
  2. Иначе (или если точек осталось мало) объект ищется заново: ключевые точки SIFT/ORB
     сопоставляются с точками эталона (kNN + тест отношения Лоу).
  3. По парам точек (эталон -> кадр) методом RANSAC оценивается гомография.
  4. Углы эталона переносятся гомографией - получаем рамку объекта.
"""
import cv2
import numpy as np


class ObjectTracker:
    def __init__(self, reference, detector="sift", use_flow=True, ratio=0.8,
                 min_inliers=10, min_flow_inliers=8, reliable_matches=25,
                 min_homography_pts=25):
        self.min_homography_pts = min_homography_pts
        self.use_flow = use_flow
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_flow_inliers = min_flow_inliers
        self.reliable_matches = reliable_matches
        if detector == "sift":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError("detector must be 'sift' or 'orb'")
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_desc = self.detector.detectAndCompute(gray, None)
        h, w = gray.shape
        self.ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        self.frame_area = None
        self.prev_gray = None
        self.ref_pts = None  # отслеживаемые точки в координатах эталона
        self.cur_pts = None  # те же точки в координатах текущего кадра

    def _make_box(self, ref_pts, cur_pts, thresh, force_homography=False):
        """Гомография по парам точек + проверка рамки. Возвращает (углы, маска инлайеров) или None."""
        if force_homography or len(ref_pts) >= self.min_homography_pts:
            H, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, thresh)
        else:
            A, mask = cv2.estimateAffinePartial2D(ref_pts, cur_pts, method=cv2.RANSAC,
                                                  ransacReprojThreshold=thresh)
            H = None if A is None else np.vstack([A, [0, 0, 1]])
        if H is None:
            return None
        corners = cv2.perspectiveTransform(self.ref_corners, H)
        area = cv2.contourArea(corners.astype(np.float32))
        if not cv2.isContourConvex(corners.astype(np.float32)) or not 20 < area < self.frame_area:
            return None
        return corners.reshape(4, 2), mask.ravel().astype(bool)

    def _detect(self, gray):
        kp, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or len(kp) < 2:
            return None
        pairs = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < self.min_inliers:
            return None
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        cur_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        res = self._make_box(ref_pts, cur_pts, 5.0, force_homography=True)
        if res is None or res[1].sum() < self.min_inliers:
            return None
        corners, inl = res
        return corners, ref_pts[inl], cur_pts[inl]

    def _flow(self, gray):
        new, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.cur_pts, None,
                                               winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, new, None,
                                                winSize=(21, 21), maxLevel=3)
        fb_err = np.linalg.norm((back - self.cur_pts).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)
        if ok.sum() < self.min_flow_inliers:
            return None
        ref_pts, cur_pts = self.ref_pts[ok], new[ok]
        res = self._make_box(ref_pts, cur_pts, 3.0)
        if res is None or res[1].sum() < self.min_flow_inliers:
            return None
        corners, inl = res
        return corners, ref_pts[inl], cur_pts[inl]

    def track(self, frame):
        """Возвращает (углы рамки 4x2 или None, число используемых точек, способ: 'flow'/'detect'/None)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_area is None:
            self.frame_area = 4.0 * gray.shape[0] * gray.shape[1]
        det = self._detect(gray)
        flow = self._flow(gray) if self.use_flow and self.cur_pts is not None else None
        if det is not None and (flow is None or len(det[1]) >= self.reliable_matches):
            result, how = det, "detect"
        elif flow is not None:
            result, how = flow, "flow"
        else:
            result, how = None, None
        self.prev_gray = gray
        if result is None:
            self.ref_pts = self.cur_pts = None
            return None, 0, None
        corners, self.ref_pts, self.cur_pts = result
        return corners, len(self.ref_pts), how


def draw_box(frame, corners, label):
    cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    x, y = corners[corners[:, 1].argmin()].astype(int)
    cv2.putText(frame, label, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, label, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
