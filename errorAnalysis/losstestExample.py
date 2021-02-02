from jd_mhqa.lossUtils import *
import torch

torch.manual_seed(11)
x = torch.rand((2,5))
y = torch.randint(0, 2, (2,5))
y[:,0] = 0
print(x)
print(y)
atloss = ATLoss()
loss = atloss.forward(logits=x, labels=y)
print(loss)

# atmloss = ATMLoss()
# loss = atloss.forward(logits=x, labels=y)
# print(loss)
#
# atploss = ATPLoss()
# loss = atploss.forward(logits=x, labels=y)
# print(loss)
#
# atpfloss = ATPFLoss()
# loss = atpfloss.forward(logits=x, labels=y)
# print(loss)