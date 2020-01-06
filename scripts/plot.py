# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#X轴，Y轴数据

ratio_M16_N16 = [
    0.660824,
    0.43718,
    0.484809,
    0.448531,
    0.420983,
    0.360199,
    0.346881,
    0.308101,
    0.391934,
]

ratio_M16_N32 = [
    0.852231,
    0.883113,
    1.003992,
    0.739881,
    0.634185,
    0.621555,
    0.580898,
    0.513336,
    0.549566,
]

ratio_M32_N32 = [
    0.825351,
    0.942688,
    1.117663,
    0.930778,
    0.946426,
    0.942446,
    0.931365,
    0.862662,
    0.898587,
]

ratio_M32_N64 = [
    0.734643,
    0.79997,
    1.091046,
    0.881671,
    0.915779,
    0.932789,
    0.917041,
    0.849883,
    0.882845,
]

ratio_M64_N64 = [
    0.523957,
    0.638479,
    1.040936,
    0.970417,
    0.970647,
    0.951553,
    0.954979,
    0.880607,
    0.928254,
]

ratio_M64_N128 = [
    0.308227,
    0.567524,
    0.860849,
    0.835883,
    0.891887,
    0.921562,
    0.928438,
    0.842934,
    0.896423,
]

size = [
    256,
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
]

plt.figure(figsize=(8,4)) 
# 创建绘图对象
plt.plot(size, ratio_M16_N16,"x-",linewidth=1, label="M16_N16 occu=50%")
plt.plot(size, ratio_M16_N32,"x-",linewidth=1, label="M16_N32 occu=50%")
plt.plot(size, ratio_M32_N32,"x-",linewidth=1, label="M32_N32 occu=100%")
plt.plot(size, ratio_M32_N64,"x-",linewidth=1, label="M32_N64 occu=100%")
plt.plot(size, ratio_M64_N64,"x-",linewidth=1, label="M64_N64 occu=100%")
plt.plot(size, ratio_M64_N128,"x-",linewidth=1, label="M64_N128 occu=100%")
# 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("Size") 
# X轴标签
plt.ylabel("Ratio")  
# Y轴标签
plt.title("Ratio by size") 
plt.legend(loc='best')
# 保存图
plt.savefig("size.jpg") 
# 显示图
plt.show()  
