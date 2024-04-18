'''
@FileName: get_histogram.py
@Description: some transforms of hisrogram
@Author: Yiwei Ren
@Copyright © 2024 Yiwei Ren
'''
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np

class Histogram():

    def __init__(self, in_root: str, out_root: str) -> None:
        self.in_root = in_root
        self.out_root = out_root

    def get_imgs(self, show: str='y') -> tuple[list[str], list[np.ndarray]]:
        filenames: list[str] = os.listdir(self.in_root)
        res: list[np.ndarray] = []
        for filename in filenames:
            filepath = os.path.join(self.in_root, filename)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            res.append(img)
            if show == 'y':
                cv.imshow(filename, img)
                cv.waitKey(0)
                cv.destroyAllWindows()
        return filenames, res
    
    def get_hist(self, save: str='y') -> None:
        '''
        To get histograms of images under `in_root` folder
        '''
        filenames, imgs = self.get_imgs('n')
        if save == 'y':
            save_path = os.path.join(self.out_root, 'histograms')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        for i in range(len(imgs)):
            filename, _ = os.path.splitext(filenames[i])
            plt.figure()
            plt.hist(imgs[i].ravel(), 256, (0,255))
            plt.title(filename)
            if save == 'y':
                plt.savefig(os.path.join(save_path, f'hist_{filename}.png'), dpi=300, transparent=True)
            plt.show()
    
    def equlize(self, save: str='n') -> None:
        '''
        To equlize the images under `in_root` folder
        '''
        filenames, imgs = self.get_imgs('y')
        if save == 'y':
            save_path = os.path.join(self.out_root, 'equlized')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        for i in range(len(imgs)):
            filename, _ = os.path.splitext(filenames[i])
            equ = cv.equalizeHist(imgs[i])
            equ = cv.normalize(equ, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            plt.figure()
            plt.hist(equ.ravel(), 256, (0,255))
            plt.title(f'equ-{filename}')
            if save == 'y':
                cv.imwrite(os.path.join(save_path, f'equ_{filename}.jpg'), equ)
                plt.savefig(os.path.join(save_path, f'hist_equ_{filename}.png'), dpi=300, transparent=True)
            cv.imshow(f'raw-{filename}', imgs[i])
            cv.imshow(f'euqlized-{filename}', equ)
            # plt.show()
            cv.waitKey(0)
            cv.destroyAllWindows()

class Enhancer(Histogram):

    def __init__(self, in_root: str, out_root: str, ) -> None:
        super().__init__(in_root, out_root)


    def __call__(self, target_file: str, reference_file: str, save: str='y') -> None:
        ''' 
        target -> reference,
        make target image matches the style of reference image
        '''
        tarname, _ = os.path.splitext(target_file)
        tar = os.path.join(self.in_root, target_file)
        ref = os.path.join(self.in_root, reference_file)
        ref_img = cv.imread(ref, cv.IMREAD_GRAYSCALE)
        tar_img = cv.imread(tar, cv.IMREAD_GRAYSCALE)
        ref_hist = cv.calcHist([ref_img], [0], None, [256], [0,256])
        ref_hist /= np.sum(ref_hist)
        target_hist = cv.calcHist([tar_img], [0], None, [256], [0,256])
        target_hist /= np.sum(target_hist)
        # calculate \sum{p_r(k)}
        ref_cdf = np.cumsum(ref_hist)
        target_cdf = np.cumsum(target_hist)
        hist_map = np.zeros(256)
        for i in range(256):
            diff = abs(target_cdf[i] - ref_cdf)
            idx = np.argmin(diff)
            hist_map[i] = idx
        height, width = tar_img.shape
        for i in range(height):
            for j in range(width):
                tar_img[i,j] = hist_map[tar_img[i,j]]
        plt.figure()
        plt.hist(tar_img.ravel(), 256, (0,255))
        plt.title(f'matched-{tarname}')
        if save == 'y':
            save_path = os.path.join(self.out_root, 'matched')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv.imwrite(os.path.join(save_path, f'matched_{tarname}.jpg'), tar_img)
            plt.savefig(os.path.join(save_path, f'hist_matched_{tarname}.png'), dpi=300, transparent=True)
        cv.imshow(f'Matched-{tarname}', tar_img)
        # plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()

class LocalEnhancer(Histogram):

    def __init__(self, in_root: str, out_root: str) -> None:
        super().__init__(in_root, out_root)
        self.save_path = os.path.join(self.out_root, 'local_enhanced')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def __call__(self, image_file: str, local_kenel: tuple[int, int]=(3, 3), save: str='y') -> None:
        img_name, _ = os.path.splitext(image_file)
        clahe = cv.createCLAHE(clipLimit=49, tileGridSize=local_kenel)
        img = cv.imread(os.path.join(self.in_root, image_file), cv.IMREAD_GRAYSCALE)
        cla_img = clahe.apply(img)
        plt.figure()
        plt.hist(cla_img.ravel(), 256, (0,255))
        plt.title(f'local-{img_name}')
        if save == 'y':
            cv.imwrite(os.path.join(self.save_path, f'local_enhanced_{img_name}.jpg'), cla_img)
            plt.savefig(os.path.join(self.save_path, f'hist_local_{img_name}.png'), dpi=300, transparent=True)
        cv.imshow(f'clahe_{img_name}', cla_img)
        cv.waitKey(0)

class Segment(Histogram):

    def __init__(self, in_root: str, out_root: str) -> None:
        super().__init__(in_root, out_root)
        self.save_path = os.path.join(self.out_root, 'segment')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def __call__(self, img_file: str, save: str='y') -> None:
        img_name, _ = os.path.splitext(img_file)
        img = cv.imread(os.path.join(self.in_root, img_file), cv.IMREAD_GRAYSCALE)
        hist = cv.calcHist([img], [0], None, [256], [0,256])
        first_peek = np.argmax(hist)
        temp = np.zeros(256, np.float32)
        for k in range(256):
            temp[k] = abs(k - first_peek) * hist[k]
        second_peek = np.argmax(temp)
        if first_peek > second_peek:
            thes = np.argmin(hist[int(second_peek):int(first_peek)])
        else:
            thes = np.argmin(hist[int(first_peek):int(second_peek)])
        print(f'segment image at threshold: {thes}')
        _, dst = cv.threshold(img, thes, 255, cv.THRESH_BINARY)
        plt.figure()
        plt.hist(dst.ravel(), 256, (0,255))
        plt.title(f'seg-{img_name}')
        if save == 'y':
            cv.imwrite(os.path.join(self.save_path, f'seg_{img_name}_at_thes_{thes}.jpg'), dst)
            plt.savefig(os.path.join(self.save_path, f'hist_seg_{img_name}.png'), dpi=300, transparent=True)
        cv.imshow(f'seg_{img_name}', dst)
        cv.waitKey(0)

if __name__ == '__main__':
    seg = LocalEnhancer('inputs', 'outputs')
    seg('test3_corrupt.pgm', save='n')
