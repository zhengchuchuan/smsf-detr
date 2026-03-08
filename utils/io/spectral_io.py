import spectral as spy
import numpy as np
import pandas as pd

channel = [450, 550, 650, 720, 750, 800, 850]

def set_byte_order(hdr_file):
    with open(hdr_file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(hdr_file, mode='w', encoding='utf-8') as f:
        for line in lines:
            if "Date" in line or "byte order" in line or "project class" in line or len(line) == 0:
                continue
            f.write(line)
        f.writelines("\nbyte order = 0")


def open_hdr_img(src_file):
    try:
        set_byte_order(src_file)
        scr_img = spy.open_image(src_file)
        arr = np.array(scr_img[:, :, :])  # 转为numpy数组
        return arr
    except IOError as e:
        print(f"无法读取文件：{src_file}，错误信息：{e}")
        return None  # 或者你可以根据需要返回其他值


def save_hdr_img(imgs, save_path, band_list, band_names=None, wavelength_units="Nanometers"):
    """Save a cube as ENVI HDR/IMG with explicit wavelengths and band names.

    Parameters
    - imgs: HxWxC ndarray (uint8/uint16/float32)
    - save_path: path to .hdr (ENVI will create .hdr + .img)
    - band_list: list of numeric wavelengths (same length as C)
    - band_names: optional list[str] for display names (same length as C)
    - wavelength_units: string for the header (default 'Nanometers')
    """
    if len(band_list) == 0:
        raise ValueError("band_list is empty")
    H, W, C = imgs.shape
    if C != len(band_list):
        raise ValueError(f"bands mismatch: imgs has {C}, band_list has {len(band_list)}")
    if band_names is not None and len(band_names) != C:
        raise ValueError(f"band_names length ({len(band_names)}) != channels ({C})")

    meta_data = {
        "samples": W,
        "lines": H,
        "bands": C,
        "header offset": 0,
        "file type": "ENVI Standard",
        "interleave": "bsq",
        "wavelength units": wavelength_units,
        "wavelength": band_list,
    }

    # 根据输入数据类型自动设置ENVI数据类型
    if imgs.dtype == np.uint16:
        meta_data["data type"] = 12  # 12 = uint16
    elif imgs.dtype == np.float32:
        meta_data["data type"] = 4   # 4 = float32
    elif imgs.dtype == np.uint8:
        meta_data["data type"] = 1   # 1 = byte
    else:
        meta_data["data type"] = 4   # 默认float32

    if band_names is None:
        # Default readable names from wavelengths
        meta_data["band names"] = [f"{w}nm" for w in band_list]
    else:
        meta_data["band names"] = band_names

    spy.envi.save_image(save_path, imgs, force=True, interleave='bsq', metadata=meta_data)


