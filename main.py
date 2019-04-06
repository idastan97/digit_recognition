import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_recognizer import recognize

# data = cv2.imread("dataset.png", cv2.IMREAD_GRAYSCALE)
# digits = np.vsplit(data, 10)

# cv2.imshow('image',digits[0])


# dgs = []
# for i in range(10):
#     rows = np.vsplit(digits[i], 5)
#     cols = []
#     for j in range(5):
#         imgs = np.hsplit(rows[j], 100)
#         for k in range(100):
#             cols.append(imgs[k])
#     dgs.append(cols) 


# recognize(dgs[8][0])

imm = cv2.imread("samples/84.jpg")
print(recognize(imm))
    
