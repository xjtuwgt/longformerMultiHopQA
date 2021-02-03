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

criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
x = torch.randn((2, 40))
y = torch.Tensor([[1, -100],[1, -100]]).long()
z = torch.zeros_like(y, dtype=torch.float).to(y)
z = z.masked_fill(y >=0, 1)
print(z)
# ls = criterion(x, y)
# print(ls)




