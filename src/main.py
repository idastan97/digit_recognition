import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_recognizer import recognize

# write filename here
imm = cv2.imread("test1.jpg")

# The results
res = recognize(imm)
print(' - The final result:')
print(*res, sep='\n')

