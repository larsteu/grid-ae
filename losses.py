import torch.nn as nn
import torch


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.eps = eps

    def forward(self, input_val, target_val):
        loss = torch.sqrt(self.mse_loss(input_val, target_val) + self.eps)
        return loss


class MSELossWithLambda(nn.Module):
    def __init__(self, l1=100):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1 = l1

    def forward(self, input_val, target_val):
        return self.mse_loss(input_val, target_val) * self.l1


class NTXLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, input_val):
        sum_val = []
        batch_size = input_val.shape[0]
        real_inputs = int(batch_size / 2)
        for i in range(real_inputs):
            upper = torch.exp(
                self.cos(input_val[i], input_val[i + real_inputs]) / self.temperature
            )
            down = torch.sum(
                torch.tensor(
                    [
                        torch.exp(self.cos(input_val[i], second_val) / self.temperature)
                        for second_val in input_val
                    ],
                    requires_grad=True,
                )
            ) - torch.exp(self.cos(input_val[i], input_val[i]) / self.temperature)
            divs = torch.div(upper, down)
            out = 0 - torch.log(divs)
            sum_val.append(out)

        return torch.sum(torch.tensor(sum_val, requires_grad=True)) / batch_size


class ContrastiveLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.eps = eps

    def forward(self, target_val, input_val, label):
        if label == 1:
            return self.mse_loss(target_val, input_val)
        else:
            return max(
                torch.tensor([0.0], requires_grad=True),
                self.eps - self.mse_loss(target_val, input_val),
            )


class CustomGridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.mse2 = torch.nn.MSELoss()

    def forward(self, target_val, generated_val, out, dis):
        r_l = self.mse(generated_val, target_val)
        k_l_1 = self.mse2(out, dis.detach())

        return r_l + k_l_1

    def MSLE(self, target, generated):
        log_dif = torch.subtract(torch.log(1 + target), torch.log(1 + generated))
        square = torch.square(log_dif)
        return torch.mean(square)
