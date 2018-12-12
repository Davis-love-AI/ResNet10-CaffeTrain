import ParseXPface
import numpy as np

p = ParseXPface.ParseXPface('/data-face/xp_data/landmarks.txt')

Faces = p.Facelandmarks_extract()
a, label = p.Croppedface(44863,0,1)
print(a.shape,label)
fp = open('parse.log', 'w')
for i in range(0,len(Faces)):
    a,label = p.Croppedface()
    #print(a.shape,label)
    fp.write(str(i) + ':' + Faces[i].imagepath + str(a.shape)+ str(label) + '\n')
fp.close()
