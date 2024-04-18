'''
@FileName: set_points.py
@Description: 手动选取配准点对
@Author: Yiwei Ren
@Copyright © 2024 Yiwei Ren
'''
import numpy as np
import cv2 as cv
import csv

class SetPoints():

    def __init__(self, image: str) -> None:
        self.times = 0
        self.img_path = image
        self.img = cv.imread(self.img_path, cv.IMREAD_COLOR)
        print(f'Image in {self.img_path}')
        print(f"Image's shape: {self.img.shape}")
        self.list = []

    def callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img,(x,y),20,(255,0,0),-1)
            cv.putText(self.img, f'{self.times} {(x, y)}', (x,y), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 5)
            self.times += 1
            self.list.append((x,y))

    def save_as_csv(self):
        with open(f'{self.img_path}_points.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.list)

    def __call__(self, *args: np.any, **kwds: np.any) -> np.any:
        cv.namedWindow(self.img_path)
        cv.setMouseCallback(self.img_path, self.callback)
        while 1:
            cv.imshow(self.img_path, self.img)
            if cv.waitKey(20) & 0xFF == 27:
                break
        cv.destroyAllWindows()
        cv.imwrite(f'{self.img_path}_changed.jpg', self.img)
        self.save_as_csv()

if __name__ == '__main__':
    a = SetPoints('stochastic_imgs/A.jpg')
    a()
    b = SetPoints('stochastic_imgs/B.jpg')
    b()