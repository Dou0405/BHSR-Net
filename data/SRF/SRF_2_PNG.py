import scipy.io as sio
import matplotlib.pyplot as plt

# 读取.mat文件
mat_file_path = 'data/predefined_SRF/landsat_srf.mat'  # 请替换为.mat文件的路径
data = sio.loadmat(mat_file_path)
# 打印.mat文件中的所有键
print("Keys in the .mat file:", data.keys())
# 假设.mat文件中包含R, G, B三个波段的光谱响应曲线
print("Shape of data['data']:", data['data'].shape)
# 需要根据具体的.mat文件内容调整提取方式
R_spectrum = data['data'][:,2]  # 替换为.mat文件中对应的变量名
G_spectrum = data['data'][:,1]  # 替换为.mat文件中对应的变量名
B_spectrum = data['data'][:,0]  # 替换为.mat文件中对应的变量名

# 绘制光谱响应曲线
plt.figure(figsize=(8, 6))
plt.plot(R_spectrum, label='Red (R) Spectrum', color='r')
plt.plot(G_spectrum, label='Green (G) Spectrum', color='g')
plt.plot(B_spectrum, label='Blue (B) Spectrum', color='b')

# 添加标签和标题
plt.xlabel('Wavelength')
plt.ylabel('Spectral Response')
plt.title('Spectral Response Curves for R, G, B Bands')

# 显示图例
plt.legend()

# 保存图像
output_image_path = 'spectral_response.png'  # 输出文件路径
plt.savefig(output_image_path)

# 显示图形
plt.show()
