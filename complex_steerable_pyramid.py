from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools

def polar_grid_transform(h, w):
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

def get_hi_lo_filter(radius, radial_val=1.0, twidth=1.0):
    """radial_valに対応する,1組のハイパスフィルタとローパスフィルタを返す

    Args:
        radius ([[float]]): 極座標の半径成分
        radial_val (float, optional): 半径しきい値. デフォルト 1.0. 
        twidth (float, optional): 周波数フィルタの遷移幅. デフォルト 1.0.

    Returns:
        hi_mask: ハイパスフィルタ
        lo_mask: ローパスフィルタ
    """
    log_rad = np.log2(radius) - np.log2(radial_val)
    hi_mask = np.clip(log_rad, -twidth, 0)
    hi_mask = np.abs(np.cos(hi_mask*np.pi/(2*twidth)))
    lo_mask = np.sqrt(1.0 - hi_mask**2)
    return hi_mask, lo_mask

def get_hi_lo_filters(radius, radial_vals, twidth=1.0):
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
        hi_mask, lo_mask = get_hi_lo_filter(radius, rad_val)
        hi_masks.append(hi_mask)
        lo_masks.append(lo_mask)
    return hi_masks, lo_masks

def get_radial_filters(hi_masks, lo_masks):
    """半径フィルタを返す

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

def get_angle_filters(angle, orientations):
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

def build_pyramid(image, filters):
    """ステアラブル・ピラミッドを構築して返す

    Args:
        image (image): 解析対象の入力画像
        filters ([image]): ハイパス・ローパスフィルタ及びサブバンドフィルタ

    Returns:
        pyramid: 構築したステアラブル・ピラミッド
    """
    h, w = image.shape[:2]
    image_dft = np.fft.fftshift(np.fft.fft2(image, s=[h+h%2, w+w%2]))[None,:,:]
    dft = image_dft * filters
    pyramid = np.fft.ifft2(np.fft.fftshift(dft, axes=(1,2)))
    return pyramid

def get_filters(size, depth, orientations, twidth=1.0):
    global h, w
    h, w = size
    angle, radius = polar_grid_transform(h+h%2, w+w%2) # 画像サイズと同じ範囲の正規化された極座標系を取得
    radius_vals = 2.0**np.arange(-depth, 1, 1)[::-1] # 周波数フィルタの半径リストを得る(多重解像度解析のレベルを得るのと同等の効果)
    hi_masks, lo_masks = get_hi_lo_filters(radius, radius_vals, twidth) # 各解像度でのハイパス・ローパスフィルタを得る
    rad_masks = get_radial_filters(hi_masks, lo_masks) # 各解像度に対応する周波数帯をマスクする半径フィルタを得る
    angle_masks = get_angle_filters(angle, orientations) # ピラミッドに方向の情報を持たせるための角度フィルタを得る

    # 上で準備した半径フィルタと角度フィルタから,ピラミッドの構築に使うフィルタリストを作成する
    # フィルタリストの最初と最後は方向の情報を持たないハイパス・ローパスフィルタで,それ以外のフィルタはサブバンドフィルタと呼ばれる
    filters = [hi_masks[0]]
    for rad_mask, angle_mask in itertools.product(rad_masks, angle_masks):
        filt = rad_mask * angle_mask / 2
        filters.append(filt)
    filters.append(lo_masks[-1])
    return filters

def complex_steerable_pyramid(image, depth, orientations, twidth=1.0):
    """入力画像からサブバンドフィルタを作成し、フィルタ群を返す

    Args:
        image (image): １チャンネルグレースケール画像, または3チャンネルカラー画像
        depth (int): サブバンドの深さ
        orientations (int): サブバンドフィルタの方向数
        twidth (int, optional): 周波数フィルタの遷移幅. デフォルト 1.0.

    Returns:
        filters: ピラミッドを構成するフィルタの集合
    """
    filters = get_filters(image.shape[:2], depth, orientations, twidth)
    if np.ndim(image) == 2:
        pyramid = build_pyramid(image, filters)
    elif np.ndim(image) == 3:
        nfilt, height, width = np.shape(filters)
        pyramid = np.zeros((nfilt, height, width, 3), np.complex128)
        for ch in range(3):
            pyramid[:,:,:,ch] = build_pyramid(image[:,:,ch], filters)
    else:
        pyramid = None
    return pyramid, filters

def _reconstruction(pyramid, filters):
    """ピラミッドとフィルタリストから画像を再構成する

    Args:
        pyramid (_type_): 複素ステアラブルピラミッド
        filters (_type_): ピラミッドを構成するフィルタリスト

    Returns:
        recon_image: ピラミッドから再構成された画像
    """
    global h, w
    recon_dft = np.zeros((h+h%2,w+w%2), dtype=np.complex128)
    first = 0
    last = len(pyramid) - 1
    for i, (pyr, filt) in enumerate(zip(pyramid, filters)):
        dft = np.fft.fftshift(np.fft.fft2(pyr))
        if (i != first) and (i != last):
            recon_dft += 2.0*dft*filt
        else:
            recon_dft += dft*filt
    recon_image = np.fft.ifft2(np.fft.ifftshift(recon_dft)).real
    return recon_image[:h,:w]

def reconstruction(pyramid, filters):
    """ピラミッドとフィルタリストから画像を再構成する

    Args:
        pyramid (_type_): 複素ステアラブルピラミッド
        filters (_type_): ピラミッドを構成するフィルタリスト

    Returns:
        recon_image: ピラミッドから再構成された画像
    """
    if np.ndim(pyramid) == 3:
        recon_image = _reconstruction(pyramid, filters)
    elif np.ndim(pyramid) == 4:
        global h, w
        recon_image = np.zeros((h,w,3), dtype=np.float64)
        for ch in range(3):
            recon_image[:,:,ch] = _reconstruction(pyramid[:,:,:,ch], filters)
    return recon_image

def pyramid_display(pyramid, depth, orientations):
    """ステアラブルピラミッドを一覧表示する

    Args:
        pyramid (_type_): 表示したいピラミッドまたはフィルタリスト
        depth (_type_): サブバンドの深さ
        orientations (_type_): サブバンドの方向数
    """
    if np.array(pyramid).dtype == np.complex128:
        pyramid = [pyr.real for pyr in pyramid]
    fig = plt.figure()
    subfigs = fig.subfigures(3,1, height_ratios=[1,depth,1])

    subfigs[0].suptitle('High Path Filter')
    ax = subfigs[0].add_subplot()
    ax.imshow(pyramid[0],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    subfigs[1].suptitle('Sub-band Filters')
    axes = subfigs[1].subplots(depth, orientations)
    for i, ax in enumerate(np.reshape(axes, (-1))):
        ax.imshow(pyramid[i+1],cmap='gray')
        ax.set_ylabel(f'level:{i//orientations}')
        ax.set_xlabel(f'orientasion:{i%orientations}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    subfigs[2].suptitle('Low Path Filter')
    ax = subfigs[2].add_subplot()
    ax.imshow(pyramid[-1],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
