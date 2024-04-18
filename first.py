import numpy as np
import cv2 as cv

def bilinear_interpolation(h: float, w: float, img: np.ndarray) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    x = h / height
    y = w / width
    lft_top_x = int(h)-1 if h >= 1 else 0
    lft_top_y = int(w)-1 if w >= 1 else 0
    lft_top_x = lft_top_x if lft_top_x <= img.shape[0] - 2 else img.shape[0] - 2
    lft_top_y = lft_top_y if lft_top_y <= img.shape[1] - 2 else img.shape[1] - 2
    f00 = img[lft_top_x][lft_top_y]
    f10 = img[lft_top_x+1][lft_top_y]
    f01 = img[lft_top_x][lft_top_y+1]
    f11 = img[lft_top_x+1][lft_top_y+1]
    r1 = (1 - x) * f00 + x * f10
    r2 = (1 - x) * f01 + x * f11
    p = (1 - y) * r1 + y * r2
    return p.astype(np.uint8)

def nearest_interpolation(h: float, w: float, img: np.ndarray) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    x = round(h) if round(h) < height else height-1
    y = round(w) if round(w) < width else width-1
    x = x if x>0 else 0
    y = y if y>0 else 0
    return img[x, y]

    
def scale(h: int, w: int, img:np.ndarray, method:str = 'bilinear') -> np.ndarray:
    if method == 'bilinear':
        interpolation = bilinear_interpolation
    elif method == 'nearest':
        interpolation = nearest_interpolation
    else:
        return cv.resize(img, (h,w), interpolation=cv.INTER_CUBIC)
    trans_matrix = np.eye(3,3, dtype=np.double)
    trans_matrix[0,0] = h / img.shape[0]
    trans_matrix[1,1] = w / img.shape[1]
    # ag = np.pi/6
    # cs = np.cos(ag)
    # sn = np.sin(ag)
    # trans_matrix = np.array([[cs, sn, 0],[-sn, cs, 0],[0,0,1]])
    # print(trans_matrix)
    _, inv_trans_matrix = cv.invert(trans_matrix)
    # print(inv_trans_matrix)
    # arr = np.array([1,2,1])
    # print(inv_trans_matrix @ arr)
    if img.ndim == 2:
        scale_img = np.zeros((h,w), dtype=np.uint8)
    else:
        scale_img = np.zeros((h,w,3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            ori_cod = inv_trans_matrix @ np.array([x,y,1])
            pixel = interpolation(ori_cod[0], ori_cod[1], img)
            scale_img[x, y] = pixel
    # print(scale_img)
    return scale_img

def gray_level(level: int, image: np.ndarray) -> np.ndarray:
    img = np.copy(image).astype(np.float32)
    p = pow(2, level) - 1
    img *= p/255
    img += 0.5
    img = img.astype(np.uint8)
    img = img.astype(np.float32)
    img *= 255/p
    img += 0.5
    return img.astype(np.uint8)

def get_mean_var(image: np.ndarray) -> tuple:
    mean = np.mean(image)
    var = np.var(image)
    return (mean, var)

def h_shear(img: np.ndarray, sv: float = 1.5) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    M = np.array([[1, sv, 0], [0, 1, 0]], dtype=np.float32)
    return cv.warpAffine(img, M,(int(height*2.5), width), borderMode=cv.BORDER_CONSTANT, borderValue=255)

def rotate(img: np.ndarray, degree: float) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    c = (width/2, height/2)
    M = cv.getRotationMatrix2D(c, degree, 1)
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    M[0, 2] += new_width / 2 - c[0]
    M[1, 2] += new_height / 2 - c[1]
    return cv.warpAffine(img, M, (int(width*1.4), int(height*1.4)), borderMode=cv.BORDER_CONSTANT, borderValue=255)

if __name__ == '__main__':
    img = cv.imread("imgs/elain1.bmp", cv.IMREAD_GRAYSCALE)
    cv.imshow('RawPicture', img)
    print(img.shape)
    # (mean, var) = get_mean_var(img)
    # print(f'mean: {mean:.4f}, variance: {var:.4f}')

    # for gl in range(1,9):
    #     gs_img = gray_level(gl, img)
    #     cv.imwrite(f'gray_level/gl_{gl}.bmp', gs_img)

    # scaled = scale(2048, 2048, img, 'cubic')
    # cv.imshow('scaled', scaled)
    cv.waitKey(0)
