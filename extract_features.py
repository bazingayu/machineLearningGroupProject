import os
import numpy as np
import cv2
from skimage import feature
from sklearn.decomposition import PCA

# root_dir = "D:/BaiduNetdiskDownload/image/CMEImages/NoCME"
# target_dir = "D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar"
# os.makedirs(target_dir, exist_ok=True)
# index = 0
# for filename1 in os.listdir((root_dir)):
#     index += 1
#     filename = os.path.join(root_dir, filename1)
#     img = cv2.imread(filename)
#     center = [img.shape[0]//2, img.shape[1]//2]
#     polar = cv2.warpPolar(img, dsize = (300, 600), center =  center, maxRadius = center[0],flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
#     polar = polar[:, 100:]
#     cv2.imwrite(os.path.join(target_dir, str(index) + ".jpg"), polar)

# root_dir = "D:/BaiduNetdiskDownload/image/CMEImages/CME_polar"
# target_dir = "D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_lbp"
# os.makedirs(target_dir, exist_ok=True)
# index = 0
#
# radius = 2
# n_point = radius * 8
# def lbp_texture(image):
#     # 使用skimage LBP方法提取图像的纹理特征
#     lbp = feature.local_binary_pattern(image,n_point,radius,'default')
#     # 统计图像直方图256维
#     max_bins = int(lbp.max() + 1)
#     # hist size:256
#     lbp_feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
#     print(lbp_feature)
#
#
#
# for filename1 in os.listdir((root_dir)):
#     filename = os.path.join(root_dir, filename1)
#     img = cv2.imread(filename, 0)
#     lbp_feature = lbp_texture(img)
#     np.save(os.path.join(target_dir, filename1[:-4] + '.npy'), lbp_feature)


root_dir = "D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar"
target_dir = "D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_hog"
os.makedirs(target_dir, exist_ok=True)
pca = PCA(n_components=100)

for filename1 in os.listdir((root_dir)):
    filename = os.path.join(root_dir, filename1)
    img = cv2.imread(filename, 0)
    img = img/255.0
    image_features = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),
                         block_norm='L2-Hys')
    np.save(os.path.join(target_dir, filename1[:-4] + '.npy'), image_features)





