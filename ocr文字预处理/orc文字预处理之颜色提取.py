
import cv2
from  tkinter import  messagebox
import numpy as np
from matplotlib import pyplot

图片=cv2.imread("F:\\11\\5.png")

if 图片 is None:
    messagebox.showerror("error","没有图片")
    exit(1)

hsv数据组 =[

    ([25,15,210],[175,240,255]),#颜色区间1
    # ([80,20,120],[145,130,130]),#颜色区间2
    # ([80,20,120],[145,130,130])
    #白色在需要另外测试可以新建一个颜色区间在hsv数据内部，提取白色，或者新建6个区间实现精准定位
]

hsv图片=cv2.cvtColor(图片,cv2.COLOR_BGR2HSV)

# 通道数据 = hsv图片[:, :, 2].ravel()
# pyplot.hist(通道数据 ,180, [0,255])
# pyplot.show() #hsv数据显示，需要结合potoshop提取数据
 #色相（Hue） 0-180、饱和度（Saturation） 0-255  和亮度（Value） 0-255 。

mask合并 = np.zeros(hsv图片.shape[:2],dtype=np.uint8)
for 初值_下限制,末值_上限制 in hsv数据组:
    初值 = np.array(初值_下限制,dtype=np.uint8)
    末值 = np.array(末值_上限制,dtype=np.uint8)
    mask提取 = cv2.inRange(hsv图片,初值,末值)
    mask合并 =cv2.bitwise_or(mask合并,mask提取)

#补全文字周边
圆化参数= cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
平滑边 = cv2.morphologyEx(mask合并,cv2.MORPH_CLOSE,圆化参数,iterations=3)

结果=cv2.bitwise_and(图片,图片,mask=平滑边)
cv2.imshow("result", 结果)
cv2.waitKey(0)

# cv2.imwrite("F:\\11\\7.png",hsv图片)
#版本要求 开发环境 py 3.12 opencv 4.10.0.84 请使用utf-8编码