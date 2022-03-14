import cv2
import numpy as np


class SlamViewer(object):
    def __init__(self):
        self.trg_img = None
        self.trg_kpts = None

    def __call__(self, src_img, src_kpts, matches):
        if self.trg_img is not None:
            if not isinstance(matches[0], cv2.DMatch):
                matches = [cv2.DMatch(m[0], m[1], m[2]) for m in matches]
            out_img = cv2.drawMatches(cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR), src_kpts,
                                      cv2.cvtColor(self.trg_img, cv2.COLOR_RGB2BGR),
                                      self.trg_kpts, matches, None, 0.5)
            cv2.imshow('matches', out_img)
            cv2.waitKey(1)
        self.trg_img = src_img
        self.trg_kpts = src_kpts
