'''
  @FileName: sixth.py
  @Description: For the sixth homework of DIP
  @Author: Yves Ren
  @FirstEdited: 16:11:16(UTC+8), Tuesday, April 09, 2024
  @Copyright © 2024 Yves Ren. All rights reserved.
'''
import numpy as np
import cv2 as cv
import os

def gaussian_noise(img: np.ndarray, mean: float=0, variance: float=1, save=True):
    nos = np.random.normal(mean, np.sqrt(variance), img.shape)
    # .astype(np.uint8)
    nos_img = img+nos
    nos_img = nom(nos_img)
    if save:
        cv.imwrite(f'outputs/gassian_imgs/m{mean}_v{variance}.jpg', nos_img)
    cv.imshow('nos', nos)
    cv.imshow('gn', nos_img)
    cv.waitKey(0)
    return nos_img

def restore(rootpath: str, outroot:str, filtertype: str='mean', ksize: tuple[int, int]=(7,7),save=False):
    img_paths = os.listdir(rootpath)
    for imgname in img_paths:
        imgp = os.path.join(rootpath, imgname)
        img = cv.imread(imgp, cv.IMREAD_GRAYSCALE)
        if filtertype == 'mean':
            fimg = cv.blur(img, ksize=ksize)
        elif filtertype == 'geometric':
            img_float = img.astype(np.float64)
            fimg = np.zeros(img.shape)
            radius = ksize[0] // 2
            for i in range(radius, img.shape[0]-radius):
                for j in range(radius, img.shape[1]-radius):
                    neighborhood = img_float[i-radius:i+radius+1, j-radius:j+radius+1]
                    fimg[i, j] = np.prod(neighborhood)**(1.0/(ksize[0]*ksize[1]))
            fimg = fimg.astype(np.uint8)
        elif filtertype == 'hormonic':
            img_float = img.astype(np.float64) + 1e-5
            rec: np.ndarray = cv.blur(1/img_float, ksize=ksize)
            rec += 1e-5
            fimg = (1 / rec).astype(np.uint8)
        elif filtertype == 'adaptive':
            img_float = img.astype(np.float64) + 1e-5
            mean = cv.blur(img_float, ksize=ksize)
            mean_sql = cv.blur(cv.multiply(img_float, img_float), ksize=ksize)
            variances = cv.subtract(mean_sql, cv.multiply(mean, mean))
            mean_total = np.mean(img_float)
            variance_total = np.var(img_float)
            fimg = np.zeros(img.shape)
            for i in range(fimg.shape[0]):
                for j in range(fimg.shape[1]):
                    frac = variance_total / variances[i, j]
                    if frac > 1:
                        frac = 1
                    fimg[i, j] = img[i, j] - frac*(img[i, j] - mean[i, j])
            fimg = fimg.astype(np.uint8)
        elif filtertype == 'median':
            fimg = cv.medianBlur(img, ksize[0])
        if save:
            name, _ = os.path.splitext(imgname)
            cv.imwrite(os.path.join(outroot, f'{filtertype}_{name}.jpg'), fimg)
        cv.imshow('f', fimg)
        cv.waitKey(0)

def impulse_noise(img: np.ndarray, pa: float=0.1, pb: float=0.1, save: bool=False):
    nos = np.random.rand(img.shape[0], img.shape[1])
    img[nos < pa] = 255
    img[nos > 1-pb] = 0
    if save:
        cv.imwrite('outputs/impulse_nos/impulse_noise.jpg', img)
    cv.imshow('ab', img)
    cv.waitKey(0)

def nom(img: np.ndarray):
    img = img.astype(np.float64)
    min_ = np.min(img)
    img -= min_
    max_ = np.max(img)
    img /= max_
    img *= 255
    img = img.astype(np.uint8)
    return img

def inv_hormonic(imgpath: str, order: float, ksize: tuple[int ,int]=(3, 3),save=False):
    image = cv.imread('outputs/impulse_nos/impulse_noise.jpg', cv.IMREAD_GRAYSCALE)
    img_float = image.astype(np.float64) + 1e-5
    son: np.ndarray = cv.blur(cv.pow(img_float, order+1), ksize=ksize)
    mom: np.ndarray = cv.blur(cv.pow(img_float, order), ksize=ksize)
    son += 1e-5
    mom += 1e-5
    filtered_image = son / mom
    filtered_image = nom(filtered_image)
    if save:
        cv.imwrite(f'outputs/restore/inv/order{order}.jpg', filtered_image)
    cv.imshow('fd', filtered_image)
    cv.waitKey(0)
    return filtered_image

def motion(img: np.ndarray, a, b, T):
    M, N = img.shape
    P, Q = 2*M, 2*N
    h = np.zeros((P, Q), dtype=np.complex64)
    for u in range(P):
        for v in range(Q):
            h[u, v] = T / (np.pi*(u*a + v*b)+1e-5) * np.sin(np.pi*(u*a + v*b)) * np.exp(-1j*np.pi*(u*a + v*b))
    return h

def fuse(img: np.ndarray):
    img = img.astype(np.float64)
    ft = motion(img, 0.1, 0.1, 1)
    img_pad: np.ndarray = cv.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], cv.BORDER_CONSTANT, value=[0])
    img_pad -= np.min(img_pad)
    img_pad /= np.max(img_pad)
    img_fft= np.fft.fftshift(np.fft.fft2(img_pad))
    g = img_fft * ft
    filtered_img = np.fft.ifft2(g)
    for i in range(filtered_img.shape[0]):
            for j in range(filtered_img.shape[1]):
                filtered_img[i, j] *= (-1)**(i+j)
    return nom(abs(filtered_img))

def motion_blur(image, size, angle):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    kernel_motion_blur = cv.warpAffine(kernel_motion_blur, cv.getRotationMatrix2D((size / 2 -0.5 , size / 2 -0.5 ), angle, 1.0), (size, size))
    blurred_image = cv.filter2D(image, -1, kernel_motion_blur)
    return blurred_image

def motion_blur_kernel(size, angle):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    kernel_motion_blur = cv.warpAffine(kernel_motion_blur, cv.getRotationMatrix2D((size / 2 -0.5 , size / 2 -0.5 ), angle, 1.0), (size, size))
    return kernel_motion_blur

def wiener_filter(img, kernel, K: float =1):
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.pad(kernel, [(0, img.shape[0]-kernel.shape[0]), (0, img.shape[1]-kernel.shape[1])], 'constant', constant_values=0)
    kernel = np.fft.fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = abs(np.fft.ifft2(dummy))
    return dummy


if __name__ == '__main__':
    img = cv.imread('lena.bmp', cv.IMREAD_GRAYSCALE)
    # gaussian_noise(img, mean=0, variance=10, save=True)
    # typess = ['mean', 'geometric', 'hormonic', 'adaptive', 'median']
    # for tp in typess:
    #     restore('outputs/gassian_imgs', 'outputs/restore_gassian', tp, save=True)
    # impulse_noise(img, save=True)
    # restore('outputs/impulse_nos', 'outputs/restore', 'median', ksize=(3,3),save=True)
    # inv_hormonic('outputs/impulse_nos/impulse_noise.jpg', order=-1.5, save=False)
    des = motion_blur(img, 5, -45)
    cv.imshow('des', des)
    cv.imwrite('outputs/wiener/weina.jpg', des)
    cv.waitKey(0)
    nosimg = gaussian_noise(des, variance=10, save=False)
    cv.imshow('desnos', nosimg)
    cv.imwrite('outputs/wiener/weina_nos.jpg', nosimg)
    cv.waitKey(0)
    img = cv.imread('outputs/wiener/weina_nos.jpg', cv.IMREAD_GRAYSCALE)
    wn = wiener_filter(img, motion_blur_kernel(5, -45), K=10)
    wn = nom(wn)
    cv.imshow('wen', wn)
    cv.imwrite('outputs/wiener/weina_restore.jpg', wn)
    cv.waitKey(0)