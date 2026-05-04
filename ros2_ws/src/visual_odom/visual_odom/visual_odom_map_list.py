from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from .landmark import Landmark

import cv2
import numpy as np


MIN_DEPTH = 400
MAX_DEPTH = 5000


class VisualOdomMap(list):
    """Verwaltet Landmarks mit automatischer Alterungsbereinigung im Hintergrund."""

    def __init__(self,
        landmarks: Optional[Iterable[Landmark]] = None,
        max_age_seconds: float = 5.0,
        cleanup_interval_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_age_seconds = float(max_age_seconds)
        self.cleanup_interval_seconds = float(cleanup_interval_seconds)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        if landmarks is not None:
            self.set_landmarks(landmarks)

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="VisualOdomMapCleanup",
            daemon=True,
        )

    

    def get_landmark(self, index: int) -> Landmark:
        with self._lock:
            return self[index]

    def add_kps(self, ps: Iterable[cv2.KeyPoint],des: np.ndarray,frame_rgb: np.ndarray, frame_depth: np.ndarray) -> None:
        for p in ps:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            depth_value = frame_depth[v, u]

            if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                b, g, r = frame_rgb[v, u]
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                self.add_landmark(Landmark(u=u, v=v, z=depth_value, des=des, color=rgb))
            

    def add_landmark(self, landmark: Landmark) -> Landmark:
        with self._lock:
            super().append(landmark)
        return landmark
    
    def append(self, landmark: Landmark) -> None:
        self.add_landmark(landmark)
    

    def extend(self, landmarks: Iterable[Landmark]) -> None:
        for landmark in landmarks:
            self.add_landmark(landmark)
    ##vielleicht nicht notwendig
    def insert_landmark(self, index: int, landmark: Landmark) -> None:
        with self._lock:
            super().insert(index, landmark)

    def get_landmarks(self) -> list[Landmark]:
        with self._lock:
            return list(self)

    def set_landmarks(self, landmarks: Iterable[Landmark]) -> None:
        with self._lock:
            super().clear()
            for landmark in landmarks:
                super().append(landmark)

    def set_landmark(self, index: int, landmark: Landmark) -> None:
        with self._lock:
            super().__setitem__(index, landmark)

    def remove_landmark(self, index: int) -> Landmark:
        with self._lock:
            return super().pop(index)

    def clear_old_landmarks(self) -> int:
        with self._lock:
            return self._remove_expired_landmarks_locked()

    def _remove_expired_landmarks_locked(self) -> int:
        now = time.monotonic()
        active_landmarks = [
            landmark
            for landmark in self
            if (now - landmark.created_at) <= self.max_age_seconds
        ]
        removed_count = len(self) - len(active_landmarks)

        if removed_count > 0:
            super().clear()
            super().extend(active_landmarks)

        return removed_count

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(self.cleanup_interval_seconds):
            with self._lock:
                self._remove_expired_landmarks_locked()

    def start(self) -> bool:
        if self._cleanup_thread.is_alive():
            return False  # Already running
        self._stop_event.clear()
        self._cleanup_thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

    def get_point_cloud(self) -> Any:
        pass