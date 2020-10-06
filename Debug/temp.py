import torch
import torch.nn as nn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class TempDebug(nn.Module):
    def __init__(self):
        super(TempDebug, self).__init__()
        self.linear_1 = nn.Linear(4,2)
        self.linear_2 = nn.Linear(2,1)

    def forward(self, input1, input2=None):
        if input2 is None:
            out_1 = self.linear_1(input1)
            out_2 = self.linear_2(out_1)
        else:
            out_11 = self.linear_1(input1)
            out_12 = self.linear_1(input2)
            out_1 = out_11 * 0.5 + out_12 * 0.5
            out_2 = self.linear_2(out_1)
        return out_2
        
if __name__ == '__main__':
    net = TempDebug()
    x1 = torch.Tensor([[1,2,3,4]])
    x2 = torch.Tensor([[4,3,2,3]])
    y = torch.Tensor([[1]])
    criterion = nn.MSELoss()
    model = TempDebug()
    out = model(x1, x2)
    loss = criterion(out, y)
    loss.backward()
    print(model.linear_1.weight.grad)