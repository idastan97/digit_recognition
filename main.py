import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_recognizer import recognize

# total_cnt = 4*10
# cnt = 0
# for j in range(10):    
#     for i in range(4):
#         print(j, i)
#         imm = cv2.imread("samples/"+str(j) + str(i+1) +".jpg")
#         plt.imshow(imm)
#         plt.show()
#         res = recognize(imm)
#         print(' - out: ', res)
#         if res[0] == j:
#             cnt+=1
# print('accuracy: ', cnt/total_cnt)


imm = cv2.imread("test7.jpg")
print(recognize(imm))

