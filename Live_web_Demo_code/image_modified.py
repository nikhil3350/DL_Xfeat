import cv2, torch, sys, numpy as np, random
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.')

# ── Define modified architecture (must match what you trained) ──────────────
class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layer(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        reduced = max(channels // reduction_ratio, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, reduced, bias=True), nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True), nn.Sigmoid(),
        )
    def forward(self, x):
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale

class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(BasicLayer(1,4,stride=1), BasicLayer(4,8,stride=2), BasicLayer(8,8,stride=1), BasicLayer(8,24,stride=2))
        self.block2 = nn.Sequential(BasicLayer(24,24,stride=1), BasicLayer(24,24,stride=1))
        self.block3 = nn.Sequential(BasicLayer(24,64,stride=2), BasicLayer(64,64,stride=1), BasicLayer(64,64,1,padding=0))
        self.block4 = nn.Sequential(BasicLayer(64,64,stride=2), BasicLayer(64,64,stride=1), BasicLayer(64,64,stride=1))
        self.block5 = nn.Sequential(BasicLayer(64,128,stride=2), BasicLayer(128,128,stride=1), BasicLayer(128,128,stride=1), BasicLayer(128,64,1,padding=0))
        self.se_fusion = SEBlock(channels=64, reduction_ratio=4)
        self.block_fusion = nn.Sequential(BasicLayer(64,64,stride=1), BasicLayer(64,64,stride=1), nn.Conv2d(64,64,1,padding=0))
        self.heatmap_head = nn.Sequential(BasicLayer(64,64,1,padding=0), BasicLayer(64,64,1,padding=0), nn.Conv2d(64,1,1), nn.Sigmoid())
        self.keypoint_head = nn.Sequential(BasicLayer(64,64,1,padding=0), BasicLayer(64,64,1,padding=0), BasicLayer(64,64,1,padding=0), nn.Conv2d(64,65,1))
        self.fine_matcher = nn.Sequential(
            nn.Linear(128,512), nn.BatchNorm1d(512,affine=False), nn.ReLU(inplace=True),
            nn.Linear(512,512), nn.BatchNorm1d(512,affine=False), nn.ReLU(inplace=True),
            nn.Linear(512,512), nn.BatchNorm1d(512,affine=False), nn.ReLU(inplace=True),
            nn.Linear(512,512), nn.BatchNorm1d(512,affine=False), nn.ReLU(inplace=True),
            nn.Linear(512,64),
        )
    def _unfold2d(self, x, ws=2):
        B,C,H,W = x.shape
        x = x.unfold(2,ws,ws).unfold(3,ws,ws).reshape(B,C,H//ws,W//ws,ws**2)
        return x.permute(0,1,4,2,3).reshape(B,-1,H//ws,W//ws)
    def forward(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        fused = self.se_fusion(x3 + x4 + x5)
        feats = self.block_fusion(fused)
        return feats, self.keypoint_head(self._unfold2d(x, ws=8)), self.heatmap_head(feats)

# ── Load modified weights ───────────────────────────────────────────────────
from modules.xfeat import XFeat
import unittest.mock as mock

# Prevent XFeat() from loading weights on init
with mock.patch('torch.load', return_value={}), \
     mock.patch.object(torch.nn.Module, 'load_state_dict', return_value=None):
    xfeat = XFeat()

# Now replace with our modified model and load correct weights
xfeat.net = XFeatModel()
state = torch.load('weights/xfeat.pt', map_location='cpu')
xfeat.net.load_state_dict(state)
xfeat.net.eval()
print('Modified model loaded successfully')

# ── Run matching ────────────────────────────────────────────────────────────
img1 = cv2.imread('assets/ref.png')
img2 = cv2.imread('assets/tgt.png')
mkpts1, mkpts2 = xfeat.match_xfeat(img1, img2)
print(f'Found {len(mkpts1)} matches')

# ── Visualize ───────────────────────────────────────────────────────────────
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

cv2.imwrite('matches_modified.png', canvas)
print('Saved to matches_modified.png')
