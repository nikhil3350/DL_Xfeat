import torch
state = torch.load(r'C:\Users\HP\Desktop\DL_Xfeat\xfeat_baseline_30000.pth', map_location='cpu')
torch.save(state['model'], 'weights/xfeat.pt')
print('done')