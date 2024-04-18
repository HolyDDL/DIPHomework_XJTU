'''
  @FileName: filter.py
  @Description: python file for filters
  @Author: Yves Ren
  @FirstEdited: 19:58:19(UTC+8), Monday, March 18, 2024
  @Copyright © 2024 Yves Ren. All rights reserved.
'''
import cv2 as cv
import numpy as np
import os

class ImageFilter():

    def __init__(self, inroot: str, outroot: str, save: bool=True) -> None:
        self.inroot = inroot
        self.outroot = outroot
        self.issave: bool = save
        if self.issave:
            self.__savedir()

    def __savedir(self):
        self.save_path = os.path.join(self.outroot, 'HighPass')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_imgs(self) -> list[tuple[str, np.ndarray]]:
        img_paths: list[str] = os.listdir(self.inroot)
        imgnames: list[str] = [os.path.splitext(i)[0] for i in img_paths]
        imgs: list[np.ndarray] = list()
        for imgpath in img_paths:
            imgs.append(cv.imread(os.path.join(self.inroot, imgpath), cv.IMREAD_GRAYSCALE))
        return list(zip(imgnames, imgs))
    
    def gauss_kernel(self, ksize: tuple[int, int]=(3, 3), sigma: float=1.) -> np.ndarray:
        kel: np.ndarray = np.ones(ksize)
        center: tuple[int, int]= (ksize[0]//2, ksize[1]//2)
        for i in range(kel.shape[0]):
            for j in range(kel.shape[1]):
                s = abs(i-center[0])
                t = abs(j-center[1])
                r2 = s**2 + t**2
                kel[i][j] = np.exp(-r2/(2*sigma**2))
        kel /= kel.sum()
        return kel

    def call_opencv(self):
        names_imgs: list[tuple[str, np.ndarray]] = self.get_imgs()
        for name, img in names_imgs:
            print(f'Processing picture: {name}')
            lap = cv.Laplacian(img, cv.CV_64F,ksize=3)
            can = cv.Canny(img, 100, 200)
            sbx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
            sby = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
            gb = cv.GaussianBlur(img, (3,3), 0)
            sp = cv.addWeighted(img, 1.5, gb, -0.5, 0)
            if self.issave:
                pathlap = os.path.join(self.save_path,f'{name}**lap.jpg')
                assert cv.imwrite(pathlap, lap), 'Failed save'
                pathlap = os.path.join(self.save_path,f'{name}**sbx=detected_vertical.jpg')
                assert cv.imwrite(pathlap, sbx), 'Failed save'
                pathlap = os.path.join(self.save_path,f'{name}**sby==detected_horizon.jpg')
                assert cv.imwrite(pathlap, sby), 'Failed save'
                pathlap = os.path.join(self.save_path,f'{name}**unsharp_masking.jpg')
                assert cv.imwrite(pathlap, sp), 'Failed save'
                pathlap = os.path.join(self.save_path,f'{name}**canny.jpg')
                assert cv.imwrite(pathlap, can), 'Failed save'
            # cv.imshow('lap', lap)
            # cv.imshow('can', can)
            # cv.imshow('sbx', sbx)
            # cv.imshow('sby', sby)
            # cv.imshow('sp', sp)
            # cv.waitKey(0)
        print('Processing done')
    
    def __call__(self, ksize: tuple[int, int]=(3, 3), kernel_type: str='gauss', sigma: float=1.):
        names_imgs: list[tuple[str, np.ndarray]] = self.get_imgs()
        if kernel_type == 'gauss':
            kernel = self.gauss_kernel(ksize=ksize, sigma=sigma)
        elif kernel_type == 'laplace':
            kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=np.float64)
            ksize=(3,3)
        elif kernel_type == 'sobel_horizon':
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float64)
            ksize=(3,3)
        elif kernel_type == 'sobel_vertical':
            kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float64)
            ksize=(3,3)
        print('Begin process')
        for name, img in names_imgs:
            print(f'Process picture: {name}')
            newimg = np.zeros(img.shape, dtype=np.float32)
            pad_img = np.pad(img, pad_width=ksize[0]//2, mode='constant', constant_values=0)
            for i in range(newimg.shape[0]):
                for j in range(newimg.shape[1]):
                    pixels = pad_img[i:i+ksize[0], j:j+ksize[1]]
                    if kernel_type == 'midvalue':
                        newimg[i, j] = np.sort(pixels.flatten())[ksize[0]*ksize[1]//2]
                    elif kernel_type == 'gauss':
                        newimg[i, j] = (pixels * kernel).sum()
                    elif kernel_type == 'canny':
                        pass
                    else:
                        # newimg[i, j] = abs((pixels * kernel).sum() + pad_img[i+1][j+1])
                        newimg[i, j] = (pixels * kernel).sum()
            newimg = newimg.astype(np.uint8)
            if self.issave:
                if kernel_type == 'gauss':
                    path = os.path.join(self.save_path,f'{name}**{kernel_type}**ksize={ksize}**sigma={sigma}.jpg')
                else:
                    path = os.path.join(self.save_path,f'{name}**{kernel_type}.jpg')
                assert cv.imwrite(path, newimg), f'Cannot Save picture: {name} !!!'
            cv.imshow(name, newimg)
            cv.waitKey(0)
            cv.destroyAllWindows()
        print('Processing done.')

if __name__ == '__main__':
    ft = ImageFilter('inputs', 'outputs', save=True)
    # ft(ksize=(3,3), kernel_type='sobel_vertical')
    ft(kernel_type='laplace')