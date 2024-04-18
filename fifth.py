'''
  @FileName: fifth.py
  @Description: The python code for spectrum filter
  @Author: Yves Ren
  @LastEdited: 09:46:09(UTC+8), Tuesday, April 02, 2024
  @Copyright © 2024 Yves Ren. All rights reserved.
'''

import numpy as np
import cv2 as cv
import os


class LPFilter():

    def __init__(self, inroot: str, outroot: str, is_save: bool=False, show: bool=True) -> None:
        self.inroot = inroot
        self.outroot = outroot
        self.is_save = is_save
        self.show = show
        if self.is_save:
            self.__savedir()
    
    def __savedir(self):
        self.save_path = os.path.join(self.outroot, 'Lowpass')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def butterworth(self, img: np.ndarray, D0: int = 10, n: int = 2) -> np.ndarray:
        M, N = img.shape
        P, Q = 2*M, 2*N
        h = np.zeros((P, Q))
        for u in range(P):
            for v in range(Q):
                d = np.sqrt((u-P/2)**2 + (v - Q/2)**2)
                h[u, v] = 1 / (1+(d/D0)**(2*n))
        self.show_(f'{D0}_butterworth', h)
        return h.real
    
    def gaussian(self, img: np.ndarray, D0: int = 10) -> np.ndarray:
        M, N = img.shape
        P, Q = 2*M, 2*N
        h = np.zeros((P, Q))
        d02 = 2*(D0**2)
        for u in range(P):
            for v in range(Q):
                d = np.sqrt((u-P/2)**2 + (v - Q/2)**2)
                h[u, v] = np.exp(-1*d**2 / d02)
        self.show_(f'{D0}_gaussian', h)
        return h.real
    
    def show_(self, name: str, array: np.ndarray) -> None:
        arr = np.copy(array)
        ptimg: np.ndarray = abs(arr)
        minval = np.min(ptimg)
        ptimg -= minval
        a = np.max(ptimg)
        ptimg /= a
        ptimg *= 255
        if self.show:
            cv.imshow(name, ptimg.astype(np.uint8))
            cv.waitKey(0)
        if self.is_save:
            cv.imwrite(os.path.join(self.save_path, f'{name}.jpg'), ptimg.astype(np.uint8))

    def __call__(self, related_imgpath: str, D0: int = 10, fliter_type: str='butterworth') -> None:
        img: np.ndarray = cv.imread(os.path.join(self.inroot, related_imgpath), cv.IMREAD_GRAYSCALE)
        
        if fliter_type == 'butterworth':
            btw_filter = self.butterworth(img, D0)
        else:
            btw_filter = self.gaussian(img, D0)
        img_pad = cv.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], cv.BORDER_CONSTANT, value=[0])
        img_shift = np.fft.fftshift(np.fft.fft2(img_pad))
        img_shift_abs = abs(img_shift)
        self.show_(f'{related_imgpath}_shift_fft', img_shift_abs)
        g = btw_filter * img_shift
        g_abs = abs(g)
        self.show_(f'{related_imgpath}_fft_multires_{fliter_type}_{D0}', g_abs)
        filtered_img = np.fft.ifft2(g).real
        for i in range(filtered_img.shape[0]):
            for j in range(filtered_img.shape[1]):
                filtered_img[i, j] *= (-1)**(i+j)
        self.show_(f'{related_imgpath}_res_{fliter_type}_{D0}', filtered_img)
        cutted_img = filtered_img[:img.shape[0], :img.shape[1]]
        self.show_(f'{related_imgpath}_cutted_{fliter_type}_{D0}', cutted_img)
        cv.destroyAllWindows()

class CalculateEnergy(LPFilter):

    def __init__(self, inroot: str, outroot: str, is_save: bool = False) -> None:
        super().__init__(inroot, outroot, is_save)

    def __call__(self, related_imgpath: str, D0: int = 10) -> float:
        img: np.ndarray = cv.imread(os.path.join(self.inroot, related_imgpath), cv.IMREAD_GRAYSCALE)
        img_pad = cv.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], cv.BORDER_CONSTANT, value=[0])
        img_fft_abs = abs(np.fft.fftshift(np.fft.fft2(img_pad)))**2
        in_sum = 0
        for i in range(img_fft_abs.shape[0]):
            for j in range(img_fft_abs.shape[1]):
                dis = np.sqrt((i-img.shape[0])**2 + (j-img.shape[1])**2)
                if dis <= D0:
                    in_sum += img_fft_abs[i][j]
        return in_sum / img_fft_abs.sum()

class HPFilter(LPFilter):
    
    def __init__(self, inroot: str, outroot: str, is_save: bool = False, show: bool = True) -> None:
        super().__init__(inroot, outroot, is_save, show)
        if self.is_save:
            self.__savedir()

    def __savedir(self):
        self.save_path = os.path.join(self.outroot, 'Highpass')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def butterworth(self, img: np.ndarray, D0: int = 10, n: int = 2) -> np.ndarray:
        M, N = img.shape
        P, Q = 2*M, 2*N
        h = np.zeros((P, Q))
        for u in range(P):
            for v in range(Q):
                d = np.sqrt((u-P/2)**2 + (v - Q/2)**2)
                h[u, v] = 1 / (1+(D0/(d+1e-7))**(2*n)+1e-7)
        self.show_(f'{D0}_butterworth', h)
        return h.real
    
    def gaussian(self, img: np.ndarray, D0: int = 10) -> np.ndarray:
        M, N = img.shape
        P, Q = 2*M, 2*N
        h = np.zeros((P, Q))
        d02 = 2*(D0**2)
        for u in range(P):
            for v in range(Q):
                d = np.sqrt((u-P/2)**2 + (v - Q/2)**2)
                h[u, v] = 1 - np.exp(-1*d**2 / d02)
        self.show_(f'{D0}_gaussian', h)
        return h.real

    def __call__(self, related_imgpath: str, D0: int = 10, fliter_type: str='butterworth') -> None:
        img: np.ndarray = cv.imread(os.path.join(self.inroot, related_imgpath), cv.IMREAD_GRAYSCALE)
        if fliter_type == 'butterworth':
            btw_filter = self.butterworth(img, D0)
        else:
            btw_filter = self.gaussian(img, D0)
        img_pad = cv.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], cv.BORDER_CONSTANT, value=[0])
        img_shift = np.fft.fftshift(np.fft.fft2(img_pad))
        img_shift_abs = abs(img_shift)
        self.show_(f'{related_imgpath}_shift_fft', img_shift_abs)
        g = btw_filter * img_shift
        g_abs = abs(g)
        self.show_(f'{related_imgpath}_fft_multires_{fliter_type}_{D0}', g_abs)
        filtered_img = np.fft.ifft2(g).real
        for i in range(filtered_img.shape[0]):
            for j in range(filtered_img.shape[1]):
                filtered_img[i, j] *= (-1)**(i+j)
        self.show_(f'{related_imgpath}_res_{fliter_type}_{D0}', filtered_img)
        cutted_img = filtered_img[:img.shape[0], :img.shape[1]]
        self.show_(f'{related_imgpath}_cutted_{fliter_type}_{D0}', cutted_img)
        cv.destroyAllWindows()

class LPandUMFilter(HPFilter):

    def __init__(self, inroot: str, outroot: str, is_save: bool = False, show: bool = True) -> None:
        super().__init__(inroot, outroot, is_save, show)
        if self.is_save:
            self.__savedir()

    def __savedir(self):
        self.save_path = os.path.join(self.outroot, 'LPandUM')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, related_imgpath: str) -> None:
        img: np.ndarray = cv.imread(os.path.join(self.inroot, related_imgpath), cv.IMREAD_GRAYSCALE)
        lap = cv.Laplacian(img, cv.CV_64F)
        lap_img = cv.convertScaleAbs(lap)
        
        bd = cv.GaussianBlur(img, (3,3), 1)
        sp = cv.addWeighted(img, 1+2, bd, -2, 0)
        # self.show_(f'{related_imgpath}_um', sp)
        if self.show:
        # self.show_(f'{related_imgpath}_lap', lap_img)
            cv.imshow(f'{related_imgpath}_lap', lap_img)
            cv.waitKey(0)
            cv.imshow(f'{related_imgpath}_um', sp)
            cv.waitKey(0)
        if self.is_save:
            cv.imwrite(os.path.join(self.save_path, f'{related_imgpath}_lap.jpg'), lap_img)
            cv.imwrite(os.path.join(self.save_path, f'{related_imgpath}_um.jpg'), sp)
if __name__ == '__main__':
    print('Low Pass Filtering...')
    ft = LPFilter('inputs', 'outputs', is_save=True, show=False)
    files = ['test1.pgm', 'test2.tif']
    Ds = [10, 50, 100]
    fliter_types = ['butterworth', 'gaussian']
    for file in files:
        for D0 in Ds:
            for fliter_type in fliter_types:
                ft(file, D0=D0, fliter_type=fliter_type)
    print('LP done')

    print('Calculating...')
    cal = CalculateEnergy('inputs', 'outputs')
    Ds = [10, 50, 100]
    files = os.listdir('inputs')
    for file in files:
        for D0 in Ds:
            rot = cal(file, D0)
            print(f"{file}'s energy in circle {D0} is: {rot*100:.2f}%")
    print('Calculating done')

    print('Hight Pass Filtering...')
    hp = HPFilter('inputs', 'outputs', is_save=True, show=False)
    files = ['test3.pgm', 'test4.bmp']
    Ds = [10, 50, 100]
    fliter_types = ['butterworth', 'gaussian']
    for file in files:
        for D0 in Ds:
            for fliter_type in fliter_types:
                hp(file, D0=D0, fliter_type=fliter_type)
    print('HP done')

    print('Laplacian and Unsharp Mask Filtering...')
    pm = LPandUMFilter('inputs', 'outputs', is_save=True, show=False)
    pm('test3.pgm')
    pm('test4.bmp')
    print('LP&UM done')