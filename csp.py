from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools

class ComplexSteerablePyramid():
    def __init__(self, image, depth, orientations, twidth=1.0):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.depth = depth
        self.orientations = orientations
        self.twidth = twidth
        self.hipath_filt = None
        self.lopath_filt = None
        self.subband_filters = None
        self.level_filters = None
        # ローパスフィルタ、ハイパスフィルタ、サブバンドフィルタを別々にメンバとして持つ
        # 必要であればゲッターメソッドを通してフィルタをまとめる
    def generate_filters(self):
        """複素ステアラブルフィルタを生成する
        """
        angle, radius = self.__polar_grid_transform(self.h, self.w)
        radius_vals = 2.0**np.arange(-self.depth, 1, 1)[::-1]
        hi_masks, lo_masks = self.__get_hi_lo_filters(radius, radius_vals, self.twidth)
        rad_masks = self.__get_radial_filters(hi_masks, lo_masks)
        angle_masks = self.__get_angle_filters(angle, self.orientations)
        self.subband_filters = []
        for rad_mask, angle_mask in itertools.product(rad_masks, angle_masks):
            filt = rad_mask * angle_mask / 2
            self.subband_filters.append(filt)
        self.hipath_filt = hi_masks[0]
        self.lopath_filt = lo_masks[-1]
        self.level_filters = lo_masks[1:]

    def __polar_grid_transform(self, h, w):
        """渡された範囲の正規極座標を返す

        Args:
            h (int): 画像の高さ
            w (int): 画像の幅

        Returns:
            angle: 極座標の角度成分
            radius: 極座標の半径成分
        """
        h_half = h//2
        w_half = w//2
        polar_x, polar_y = np.meshgrid(np.arange(-w_half, w_half)/w_half, np.arange(-h_half, h_half)/h_half)
        angle = np.arctan2(polar_y, polar_x)
        radius = np.sqrt(polar_x**2 + polar_y**2)
        radius[h_half][w_half] = radius[h_half][w_half-1]
        return angle, radius
    def __get_hi_lo_filters(self, radius, radial_vals, twidth=1.0):
        """ピラミッドのレベルに対応するハイパスフィルタとローパスフィルタのリストを返す

        Args:
            radius ([[float]]): 極座標の半径成分
            radial_vals ([float]): 半径しきい値リスト
            twidth (float, optional): 周波数フィルタの遷移幅. デフォルト 1.0.

        Returns:
            hi_masks: ハイパスフィルタのリスト
            lo_masks: ローパスフィルタのリスト
        """
        hi_masks = []
        lo_masks = []
        for rad_val in radial_vals:
            log_rad = np.log2(radius) - np.log2(rad_val)
            hi_mask = np.clip(log_rad, -twidth, 0)
            hi_mask = np.abs(np.cos(hi_mask*np.pi/(2*twidth)))
            lo_mask = np.sqrt(1.0 - hi_mask**2)
            hi_masks.append(hi_mask)
            lo_masks.append(lo_mask)
        return hi_masks, lo_masks

    def __get_radial_filters(self, hi_masks, lo_masks):
        """半径フィルタのリストを返す

        Args:
            hi_masks ([hi_mask]): ハイパスフィルタのリスト
            lo_masks ([lo_mask]): ローパスフィルタのリスト

        Returns:
            rad_masks: 半径フィルタのリスト
        """
        rad_masks = []
        for i in range(1, len(hi_masks)):
            rad_mask = hi_masks[i] * lo_masks[i-1]
            rad_masks.append(rad_mask)
        return rad_masks

    def __get_angle_filters(self, angle, orientations):
        """方向フィルタのリストを返す

        Args:
            angle ([[float32]]): 極座標の角度成分

        Returns:
            angle_masks: 方向フィルタのリスト
        """
        angle_masks = []
        for b in range(orientations):
            order = orientations - 1
            const = np.power(2, (2*order)) * np.power(factorial(order), 2) / (orientations*factorial(2*order))
            tmp_angle = np.mod(np.pi + angle - np.pi*b/orientations, 2*np.pi) - np.pi
            angle_mask = 2*np.sqrt(const) * np.power(np.cos(tmp_angle), order) * (np.abs(tmp_angle) < np.pi/2)
            angle_masks.append(angle_mask)
        return angle_masks

    def get_hipath_filt(self):
        return self.hipath_filt
    def get_lopath_filt(self):
        return self.lopath_filt
    def get_subband_filters(self):
        return self.subband_filters
    def get_level_filters(self):
        return self.level_filters