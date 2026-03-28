import cv2, torch, sys, numpy as np, random
sys.path.insert(0, '.')
from modules.xfeat import XFeat

xfeat = XFeat()
img1 = cv2.imread('assets/ref.png')
img2 = cv2.imread('assets/tgt.png')
mkpts1, mkpts2 = xfeat.match_xfeat(img1, img2)
print(f'Found {len(mkpts1)} matches')

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
canvas = np.zeros((max(h1,h2), w1+w2, 3), dtype='uint8')
canvas[:h1, :w1] = img1
canvas[:h2, w1:] = img2

for i in range(0, len(mkpts1), 3):
    pt1 = (int(mkpts1[i][0]), int(mkpts1[i][1]))
    pt2 = (int(mkpts2[i][0]) + w1, int(mkpts2[i][1]))
    color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
    cv2.line(canvas, pt1, pt2, color, 1)
    cv2.circle(canvas, pt1, 3, color, -1)
    cv2.circle(canvas, pt2, 3, color, -1)

cv2.imwrite('matches_output_baseline.png', canvas)
print('Saved to matches_output.png')