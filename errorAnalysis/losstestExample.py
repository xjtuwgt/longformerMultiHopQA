from jd_mhqa.lossUtils import *
import torch

torch.manual_seed(16)
x = torch.rand((2, 7))
y = torch.randint(0,2,(2,7))
y[:,0] = 0
x_p_mask = (y == 1).float()*1
mask = x_p_mask.clone()
mask[:,0]=1
x = x + x_p_mask
x[:,0] = 2
print(x)


# x = torch.Tensor([[5, 20, 20, 0.2, 0.2],[5, 20, 20, 0.35, 0.3]])
# y = torch.Tensor([[0,1,1,0,0],[0,1,1,0,0]])
print(x)
print(y)
# y[:,0] = 0
# print(x)
# print(y)
atloss = ATLoss()
loss = atloss.forward(logits=x, labels=y)
print(loss)


#
atmloss = ATMLoss()
loss = atmloss.forward(logits=x, labels=y)
print(loss)
print(x)
#
# # y = adaptive_threshold_prediction(logits=x, number_labels=2, type='topk')
# # print(y)
# #
atploss = ATPLoss()
loss = atploss.forward(logits=x, labels=y)
print(loss)
# #
atpfloss = ATPFLoss()
loss = atpfloss.forward(logits=x, labels=y, mask=mask)
print(loss)

# criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
# x = torch.randn((2, 40))
# y = torch.Tensor([[1, -100],[1, -100]]).long()
# z = torch.zeros_like(y, dtype=torch.float).to(y)
# z = z.masked_fill(y >=0, 1)
# print(z)
# ls = criterion(x, y)
# print(ls)
z = adaptive_threshold_prediction(logits=x, number_labels=-1, type='or')
print(z)



