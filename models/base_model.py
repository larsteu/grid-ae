import os
from abc import ABC

import torch


class BaseModel(ABC):
    def save_model(self, optimizer, path, filename):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(checkpoint, os.path.join(path, filename))

    def load_model(self, optimizer, lr, path, filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(
            os.path.join(path, filename),
        )
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
