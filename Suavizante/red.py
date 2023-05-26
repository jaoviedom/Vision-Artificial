import cv2
import numpy as np
import pandas as pd

img1 = cv2.imread('./images/suavizante/Manzana/1.jpeg')
img2 = cv2.imread('./images/suavizante/Manzana/2.jpeg')
img3 = cv2.imread('./images/suavizante/Manzana/3.jpeg')
img4 = cv2.imread('./images/suavizante/Manzana/4.jpeg')
img5 = cv2.imread('./images/suavizante/Manzana/5.jpeg')
img6 = cv2.imread('./images/suavizante/Manzana/6.jpeg')
img7 = cv2.imread('./images/suavizante/Manzana/7.jpeg')
img8 = cv2.imread('./images/suavizante/Manzana/8.jpeg')
img9 = cv2.imread('./images/suavizante/Manzana/9.jpeg')
bordes1 = cv2.Canny(img1, 135, 150)
bordes2 = cv2.Canny(img2, 135, 150)
bordes3 = cv2.Canny(img3, 135, 150)
bordes4 = cv2.Canny(img4, 135, 150)
bordes5 = cv2.Canny(img5, 135, 150)
bordes6 = cv2.Canny(img6, 135, 150)
bordes7 = cv2.Canny(img7, 135, 150)
bordes8 = cv2.Canny(img8, 135, 150)
bordes9 = cv2.Canny(img9, 135, 150)

# cv2.imshow("Bordes", bordes)

num_kpt = 1000
# Declaramos el objeto
orb = cv2.ORB_create(num_kpt)
# Extraemos la info de la img
# keypoint1, descriptor1 = orb.detectAndCompute(img, None)
keypoint1, descriptor1 = orb.detectAndCompute(bordes1, None)
keypoint2, descriptor2 = orb.detectAndCompute(bordes2, None)
keypoint3, descriptor3 = orb.detectAndCompute(bordes3, None)
keypoint4, descriptor4 = orb.detectAndCompute(bordes4, None)
keypoint5, descriptor5 = orb.detectAndCompute(bordes5, None)
keypoint6, descriptor6 = orb.detectAndCompute(bordes6, None)
keypoint7, descriptor7 = orb.detectAndCompute(bordes7, None)
keypoint8, descriptor8 = orb.detectAndCompute(bordes8, None)
keypoint9, descriptor9 = orb.detectAndCompute(bordes9, None)

# print(keypoint1, descriptor1)
npKeyPoint1 = np.array(keypoint1)
npDescriptor1 = np.array(descriptor1)
# print(npKeyPoint1, npDescriptor1)
# print(npKeyPoint1.shape)
# print(npDescriptor1.shape)

############################################################################
import pandas as pd

df_descr1 = pd.DataFrame(descriptor1)
df_descr1.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints1 = pd.DataFrame(keypoint1)
df_descr1['Keypoint'] = df_keypoints1
print(df_descr1)

df_descr2 = pd.DataFrame(descriptor2)
df_descr2.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints2 = pd.DataFrame(keypoint2)
df_descr2['Keypoint'] = df_keypoints2

df_descr3 = pd.DataFrame(descriptor3)
df_descr3.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints3 = pd.DataFrame(keypoint3)
df_descr3['Keypoint'] = df_keypoints3

df_descr4 = pd.DataFrame(descriptor4)
df_descr4.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints4 = pd.DataFrame(keypoint4)
df_descr4['Keypoint'] = df_keypoints4

df_descr5 = pd.DataFrame(descriptor5)
df_descr5.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints5 = pd.DataFrame(keypoint5)
df_descr5['Keypoint'] = df_keypoints5

df_descr6 = pd.DataFrame(descriptor6)
df_descr6.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints6 = pd.DataFrame(keypoint6)
df_descr6['Keypoint'] = df_keypoints6

df_descr7 = pd.DataFrame(descriptor7)
df_descr7.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints7 = pd.DataFrame(keypoint7)
df_descr7['Keypoint'] = df_keypoints7

df_descr8 = pd.DataFrame(descriptor8)
df_descr8.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints8 = pd.DataFrame(keypoint8)
df_descr8['Keypoint'] = df_keypoints8

df_descr9 = pd.DataFrame(descriptor9)
df_descr9.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
df_keypoints9 = pd.DataFrame(keypoint9)
df_descr9['Keypoint'] = df_keypoints9

df = pd.concat([df_descr1, df_descr2, df_descr3, df_descr4, df_descr5, df_descr6, df_descr7, df_descr8, df_descr9])

print(df.shape)
print(df.head())

print(df.info())
# print(cv2.KeyPoint.convert(keypoint1))
df_key = pd.DataFrame(cv2.KeyPoint.convert(keypoint1))
print(df_key)

# df.to_csv('./Dataset_suavizante.csv', index=False)