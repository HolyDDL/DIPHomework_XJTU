'''
@FileName: calculate.py
@Description: 基于第一次手动选取的配准点对计算转移矩阵, 并输出仿射变换后图像和对比图像
@Author: Yiwei Ren
@Copyright © 2024 Yiwei Ren
'''
import cv2 as cv
import numpy as np
import csv

class Calculate_H():

    def __init__(self, A_path: str, B_path: str) -> None:
        self.A_img_path = A_path
        self.B_img_path = B_path
        self.A_img = cv.imread(self.A_img_path, cv.IMREAD_COLOR)
        self.B_img = cv.imread(self.B_img_path, cv.IMREAD_COLOR)
    
    def read_raw_matrix(self, img_path: str):
        img_csv = f'{img_path}_points.csv'
        ls = []
        with open(img_csv, 'r') as f:
            reader = csv.reader(f)
            for x, y in reader:
                x = eval(x)
                y = eval(y)
                ls.append([x,y,1])
        raw_matrix = np.array(ls, dtype=np.float64).T
        return raw_matrix
    
    def show(self):
        H = self.__call__()
        np.set_printoptions(precision=4, suppress=True)
        print(f'Transform matrix: \n{H}')
        af = H[:2, :]
        dst = cv.warpAffine(self.A_img, af, (self.B_img.shape[1], self.B_img.shape[0]),borderMode=cv.BORDER_CONSTANT, borderValue=[255,255,255])
        cv.imshow('Raw_B', self.B_img)
        cv.imshow('Changed_B', dst)
        sub = dst - self.B_img
        cv.imshow('Sub', sub)
        cv.imwrite(f'{self.A_img_path}_transformed.jpg', dst)
        cv.imwrite(f'{self.A_img_path}_transform_sub.jpg', sub)
        cv.waitKey(0)
    
    def __call__(self, *args: np.any, **kwds: np.any) -> np.any:
        P = self.read_raw_matrix(self.A_img_path)
        Q = self.read_raw_matrix(self.B_img_path)
        return Q @ P.T @ np.linalg.inv(P @ P.T)
    
if __name__ == '__main__':
    ct = Calculate_H('stochastic_imgs/A.jpg', 'stochastic_imgs/B.jpg')
    ct.show()