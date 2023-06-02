import cv2
import numpy as np
import pandas as pd

img11 = cv2.imread('./images/suavizante/Manzana/1.jpeg')
img12 = cv2.imread('./images/suavizante/Manzana/2.jpeg')
img13 = cv2.imread('./images/suavizante/Manzana/3.jpeg')
img14 = cv2.imread('./images/suavizante/Manzana/4.jpeg')
img15 = cv2.imread('./images/suavizante/Manzana/5.jpeg')
img16 = cv2.imread('./images/suavizante/Manzana/6.jpeg')
img17 = cv2.imread('./images/suavizante/Manzana/7.jpeg')
img18 = cv2.imread('./images/suavizante/Manzana/8.jpeg')
img19 = cv2.imread('./images/suavizante/Manzana/9.jpeg')
img21 = cv2.imread('./images/suavizante/Primavera/1.jpeg')
img22 = cv2.imread('./images/suavizante/Primavera/2.jpeg')
img23 = cv2.imread('./images/suavizante/Primavera/3.jpeg')
img24 = cv2.imread('./images/suavizante/Primavera/4.jpeg')
img25 = cv2.imread('./images/suavizante/Primavera/5.jpeg')
img26 = cv2.imread('./images/suavizante/Primavera/6.jpeg')
img27 = cv2.imread('./images/suavizante/Primavera/7.jpeg')
img28 = cv2.imread('./images/suavizante/Primavera/8.jpeg')
img29 = cv2.imread('./images/suavizante/Primavera/9.jpeg')
bordes11 = cv2.Canny(img11, 135, 150)
bordes12 = cv2.Canny(img12, 135, 150)
bordes13 = cv2.Canny(img13, 135, 150)
bordes14 = cv2.Canny(img14, 135, 150)
bordes15 = cv2.Canny(img15, 135, 150)
bordes16 = cv2.Canny(img16, 135, 150)
bordes17 = cv2.Canny(img17, 135, 150)
bordes18 = cv2.Canny(img18, 135, 150)
bordes19 = cv2.Canny(img19, 135, 150)
bordes21 = cv2.Canny(img21, 135, 150)
bordes22 = cv2.Canny(img22, 135, 150)
bordes23 = cv2.Canny(img23, 135, 150)
bordes24 = cv2.Canny(img24, 135, 150)
bordes25 = cv2.Canny(img25, 135, 150)
bordes26 = cv2.Canny(img26, 135, 150)
bordes27 = cv2.Canny(img27, 135, 150)
bordes28 = cv2.Canny(img28, 135, 150)
bordes29 = cv2.Canny(img29, 135, 150)

num_kpt = 1000
# Declaramos el objeto
orb = cv2.ORB_create(num_kpt)
# Extraemos la info de la img
# keypoint1, descriptor1 = orb.detectAndCompute(img, None)
keypoint11, descriptor11 = orb.detectAndCompute(bordes11, None)
keypoint12, descriptor12 = orb.detectAndCompute(bordes12, None)
keypoint13, descriptor13 = orb.detectAndCompute(bordes13, None)
keypoint14, descriptor14 = orb.detectAndCompute(bordes14, None)
keypoint15, descriptor15 = orb.detectAndCompute(bordes15, None)
keypoint16, descriptor16 = orb.detectAndCompute(bordes16, None)
keypoint17, descriptor17 = orb.detectAndCompute(bordes17, None)
keypoint18, descriptor18 = orb.detectAndCompute(bordes18, None)
keypoint19, descriptor19 = orb.detectAndCompute(bordes19, None)
keypoint21, descriptor21 = orb.detectAndCompute(bordes21, None)
keypoint22, descriptor22 = orb.detectAndCompute(bordes22, None)
keypoint23, descriptor23 = orb.detectAndCompute(bordes23, None)
keypoint24, descriptor24 = orb.detectAndCompute(bordes24, None)
keypoint25, descriptor25 = orb.detectAndCompute(bordes25, None)
keypoint26, descriptor26 = orb.detectAndCompute(bordes26, None)
keypoint27, descriptor27 = orb.detectAndCompute(bordes27, None)
keypoint28, descriptor28 = orb.detectAndCompute(bordes28, None)
keypoint29, descriptor29 = orb.detectAndCompute(bordes29, None)

############################################################################

df_descr11 = pd.DataFrame(descriptor11)
df_descr11.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = []; yPoints= []; angles = []; octaves = []; responses = []; sizes = []; classes = []

for kp in keypoint11:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr11['X'] = df_xPoints
df_descr11['Y'] = df_yPoints
df_descr11['Angles'] = df_angles
df_descr11['Octaves'] = df_octaves
df_descr11['Responses'] = df_responses
df_descr11['Sizes'] = df_sizes
df_descr11['Classes'] = df_classes

df_descr12 = pd.DataFrame(descriptor12)
df_descr12.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint12:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr12['X'] = df_xPoints
df_descr12['Y'] = df_yPoints
df_descr12['Angles'] = df_angles
df_descr12['Octaves'] = df_octaves
df_descr12['Responses'] = df_responses
df_descr12['Sizes'] = df_sizes
df_descr12['Classes'] = df_classes

df_descr13 = pd.DataFrame(descriptor13)
df_descr13.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint13:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr13['X'] = df_xPoints
df_descr13['Y'] = df_yPoints
df_descr13['Angles'] = df_angles
df_descr13['Octaves'] = df_octaves
df_descr13['Responses'] = df_responses
df_descr13['Sizes'] = df_sizes
df_descr13['Classes'] = df_classes

df_descr14 = pd.DataFrame(descriptor14)
df_descr14.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint14:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr14['X'] = df_xPoints
df_descr14['Y'] = df_yPoints
df_descr14['Angles'] = df_angles
df_descr14['Octaves'] = df_octaves
df_descr14['Responses'] = df_responses
df_descr14['Sizes'] = df_sizes
df_descr14['Classes'] = df_classes

df_descr15 = pd.DataFrame(descriptor15)
df_descr15.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint15:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr15['X'] = df_xPoints
df_descr15['Y'] = df_yPoints
df_descr15['Angles'] = df_angles
df_descr15['Octaves'] = df_octaves
df_descr15['Responses'] = df_responses
df_descr15['Sizes'] = df_sizes
df_descr15['Classes'] = df_classes

df_descr16 = pd.DataFrame(descriptor16)
df_descr16.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint16:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr16['X'] = df_xPoints
df_descr16['Y'] = df_yPoints
df_descr16['Angles'] = df_angles
df_descr16['Octaves'] = df_octaves
df_descr16['Responses'] = df_responses
df_descr16['Sizes'] = df_sizes
df_descr16['Classes'] = df_classes

df_descr17 = pd.DataFrame(descriptor17)
df_descr17.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint17:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr17['X'] = df_xPoints
df_descr17['Y'] = df_yPoints
df_descr17['Angles'] = df_angles
df_descr17['Octaves'] = df_octaves
df_descr17['Responses'] = df_responses
df_descr17['Sizes'] = df_sizes
df_descr17['Classes'] = df_classes

df_descr18 = pd.DataFrame(descriptor18)
df_descr18.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint18:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr18['X'] = df_xPoints
df_descr18['Y'] = df_yPoints
df_descr18['Angles'] = df_angles
df_descr18['Octaves'] = df_octaves
df_descr18['Responses'] = df_responses
df_descr18['Sizes'] = df_sizes
df_descr18['Classes'] = df_classes

df_descr19 = pd.DataFrame(descriptor19)
df_descr19.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint19:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr19['X'] = df_xPoints
df_descr19['Y'] = df_yPoints
df_descr19['Angles'] = df_angles
df_descr19['Octaves'] = df_octaves
df_descr19['Responses'] = df_responses
df_descr19['Sizes'] = df_sizes
df_descr19['Classes'] = df_classes

#----------------------------------------------------------------

df_descr21 = pd.DataFrame(descriptor21)
df_descr21.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = []; yPoints= []; angles = []; octaves = []; responses = []; sizes = []; classes = []

for kp in keypoint21:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr21['X'] = df_xPoints
df_descr21['Y'] = df_yPoints
df_descr21['Angles'] = df_angles
df_descr21['Octaves'] = df_octaves
df_descr21['Responses'] = df_responses
df_descr21['Sizes'] = df_sizes
df_descr21['Classes'] = df_classes

df_descr22 = pd.DataFrame(descriptor22)
df_descr22.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint22:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr22['X'] = df_xPoints
df_descr22['Y'] = df_yPoints
df_descr22['Angles'] = df_angles
df_descr22['Octaves'] = df_octaves
df_descr22['Responses'] = df_responses
df_descr22['Sizes'] = df_sizes
df_descr22['Classes'] = df_classes

df_descr23 = pd.DataFrame(descriptor23)
df_descr23.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint23:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr23['X'] = df_xPoints
df_descr23['Y'] = df_yPoints
df_descr23['Angles'] = df_angles
df_descr23['Octaves'] = df_octaves
df_descr23['Responses'] = df_responses
df_descr23['Sizes'] = df_sizes
df_descr23['Classes'] = df_classes

df_descr24 = pd.DataFrame(descriptor24)
df_descr24.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint24:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr24['X'] = df_xPoints
df_descr24['Y'] = df_yPoints
df_descr24['Angles'] = df_angles
df_descr24['Octaves'] = df_octaves
df_descr24['Responses'] = df_responses
df_descr24['Sizes'] = df_sizes
df_descr24['Classes'] = df_classes

df_descr25 = pd.DataFrame(descriptor25)
df_descr25.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint25:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr25['X'] = df_xPoints
df_descr25['Y'] = df_yPoints
df_descr25['Angles'] = df_angles
df_descr25['Octaves'] = df_octaves
df_descr25['Responses'] = df_responses
df_descr25['Sizes'] = df_sizes
df_descr25['Classes'] = df_classes

df_descr26 = pd.DataFrame(descriptor26)
df_descr26.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint16:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr26['X'] = df_xPoints
df_descr26['Y'] = df_yPoints
df_descr26['Angles'] = df_angles
df_descr26['Octaves'] = df_octaves
df_descr26['Responses'] = df_responses
df_descr26['Sizes'] = df_sizes
df_descr26['Classes'] = df_classes

df_descr27 = pd.DataFrame(descriptor27)
df_descr27.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint27:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr27['X'] = df_xPoints
df_descr27['Y'] = df_yPoints
df_descr27['Angles'] = df_angles
df_descr27['Octaves'] = df_octaves
df_descr27['Responses'] = df_responses
df_descr27['Sizes'] = df_sizes
df_descr27['Classes'] = df_classes

df_descr28 = pd.DataFrame(descriptor28)
df_descr28.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint28:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr28['X'] = df_xPoints
df_descr28['Y'] = df_yPoints
df_descr28['Angles'] = df_angles
df_descr28['Octaves'] = df_octaves
df_descr28['Responses'] = df_responses
df_descr28['Sizes'] = df_sizes
df_descr28['Classes'] = df_classes

df_descr29 = pd.DataFrame(descriptor29)
df_descr29.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
xPoints = yPoints = angles = octaves = responses = sizes = classes = []

for kp in keypoint29:
  x,y = kp.pt
  xPoints.append(x)
  yPoints.append(y)
  angles.append(kp.angle)
  octaves.append(kp.octave)
  responses.append(kp.response)
  sizes.append(kp.size)
  classes.append(kp.class_id)

df_xPoints = pd.DataFrame(xPoints)
df_yPoints = pd.DataFrame(yPoints)
df_angles = pd.DataFrame(angles)
df_octaves = pd.DataFrame(octaves)
df_responses = pd.DataFrame(responses)
df_sizes = pd.DataFrame(sizes)
df_classes = pd.DataFrame(classes)

df_descr29['X'] = df_xPoints
df_descr29['Y'] = df_yPoints
df_descr29['Angles'] = df_angles
df_descr29['Octaves'] = df_octaves
df_descr29['Responses'] = df_responses
df_descr29['Sizes'] = df_sizes
df_descr29['Classes'] = df_classes

df_manzana = pd.concat([df_descr11, df_descr12, df_descr13, df_descr14, df_descr15, df_descr16, df_descr17, df_descr18, df_descr19])
suavizante_uno = np.ones(df_manzana.shape[0]) # Manzana
df_manzana['Tipo'] = pd.DataFrame(suavizante_uno)

df_primavera = pd.concat([df_descr21, df_descr22, df_descr23, df_descr24, df_descr25, df_descr26, df_descr27, df_descr28, df_descr29])
suavizante_uno = 2 * np.ones(df_primavera.shape[0]) # Primavera
df_primavera['Tipo'] = pd.DataFrame(suavizante_uno)

df = pd.concat([df_manzana, df_primavera])
print(df.shape)
# print(df.info())


df.to_csv('./Dataset_suavizante.csv', index=False)
