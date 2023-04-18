import matplotlib.pyplot as plt
import numpy as np

with open('ckpt/res.txt', 'r') as rd:
    data = rd.readlines()
rd.close()
x = []
y = []
for line in data:
    x.append(float(line.split(', ')[0]))
    y.append(float(line.split(', ')[1].strip()))
plt.figure()
x_axis = np.arange(0.12, 0.52, 0.05)
y_axis = np.arange(0.8, 1, 0.05)
plt.xticks(x_axis)
plt.yticks(y_axis)

plt.xlabel('bit per pixel (bpp)')
plt.ylabel('ssim (RGB)')

plt.plot(x, y)
plt.show()

# a = np.arange(10,100)
# b = np.arange(40,130)
#
# print(a[::5])
#
# # 设置x/y轴尺度
# plt.xticks(a[::5])
# plt.yticks(b[::10])
#
# # 传入xy轴参数，默认为y轴;label 指定图例名称
# plt.plot(a,label="a",linestyle="--",color="blue")
# plt.plot(b,label="b",color="green")
#
# plt.legend(loc="best")  # 设置图例位置
#
# # 指定xy轴 名称
# plt.ylabel("This is Y")
# plt.xlabel("This is X")
#
# # 保存图像 默认png格式，其中dpi指图片质量
# plt.savefig("05.png", dpi=600)
#
# plt.show()  # 展示图片

