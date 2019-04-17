import cv2
import numpy as np
import matplotlib.pyplot as plt
from digit_recognizer import recognize

# write filename here
imm = cv2.imread("test9.jpg")

# The results
print(*recognize(imm), sep='\n')

