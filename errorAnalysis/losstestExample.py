from jd_mhqa.lossUtils import ATLoss
import torch

torch.manual_seed(10)
x = torch.rand((3,10))
y = torch.randint(0, 2, (3,10))
y[:,0] = 0
print(x)
print(y)
atloss = ATLoss()
loss = atloss.forward(logits=x, labels=y)
print(loss)