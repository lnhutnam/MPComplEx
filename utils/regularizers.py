from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


def pick_embregularizer(regularizer: str, emb_reg):
    if regularizer == "N3":
        emb_reg = N3(emb_reg)
    elif regularizer == "N3_eps":
        emb_reg = N3_eps(emb_reg)
    elif regularizer == "L2":
        emb_reg = L2(emb_reg)
    elif regularizer == "ER":
        emb_reg = ER(emb_reg)
    elif regularizer == "Fro":
        emb_reg = Fro(emb_reg)
    elif regularizer == "DURA":
        emb_reg = DURA(emb_reg)
    elif regularizer == "DURAW":
        emb_reg = DURA_W(emb_reg)
    elif regularizer == "DURA_RESCAL":
        emb_reg = DURA_RESCAL(emb_reg)
    elif regularizer == "DURA_RESCALW":
        emb_reg = DURA_RESCAL_W(emb_reg)
    else:
        raise NotImplementedError(f"Regularizer {regularizer} not implemented.")

    return emb_reg


def pick_timeregularizer(regularizer: str, time_reg):
    if regularizer == "Lambda3":
        time_reg = Lambda3(time_reg)
    elif regularizer == "Lambda3_eps":
        time_reg = Lambda3_eps(time_reg)
    elif regularizer == "Lambda32":
        time_reg = Lambda32(time_reg)
    elif regularizer == "Linear3":
        time_reg = Linear3(time_reg)
    elif regularizer == "Spiral3":
        time_reg = Spiral3(time_reg)
    elif regularizer == "Temporal_MSE_Regularizer":
        time_reg = Temporal_MSE_Regularizer(time_reg)
    else:
        raise NotImplementedError(f"Regularizer {regularizer} not implemented.")
    return time_reg


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor], time=None):
        pass


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class N3_eps(Regularizer):
    def __init__(self, weight: float, eps: float = 1e-5):
        super(N3_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f + self.eps) ** 3)
        return norm / factors[0].shape[0]


class L2(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class L2_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(self, weight: float, eps: float = 1e-5):
        super(L2_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f + self.eps) ** 3)
        return norm / factors[0].shape[0]


class L1(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f))
        return norm / factors[0].shape[0]


class L1_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(self, weight: float, eps: float = 1e-5):
        super(L1_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f + self.eps))
        return norm / factors[0].shape[0]


class NA(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2) ** 3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Lambda3_eps(Regularizer):
    def __init__(self, weight: float, eps: float = 1e-5):
        super(Lambda3_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2 + self.eps) ** 3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Lambda32(Regularizer):
    def __init__(self, weight: float):
        super(Lambda32, self).__init__()
        self.weight = weight

    def forward(self, factor):
        tot = 0
        if factor is not None:
            for f in factor:
                rank = int(f.shape[1] / 2)
                ddiff = f[1:] - f[:-1]
                diff = torch.sqrt((ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2)) ** 4
                tot = tot + self.weight * (torch.sum(diff))
            return tot / factor[0].shape[0]
        return 0


class Linear3(Regularizer):
    # Refs: https://github.com/soledad921/TeLM/blob/main/tkbc/regularizers.py
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W=None):
        rank = int(factor.shape[1] / 2)
        if W is not None:
            ddiff = factor[1:] - factor[:-1] - W.weight[: rank * 2].t()
        else:
            ddiff = factor[1:] - factor[:-1]
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2) ** 3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Linear3_eps(Regularizer):
    # Refs: https://github.com/soledad921/TeLM/blob/main/tkbc/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(Linear3_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factor, W=None):
        rank = int(factor.shape[1] / 2)
        if W is not None:
            ddiff = factor[1:] - factor[:-1] - W.weight[: rank * 2].t()
        else:
            ddiff = factor[1:] - factor[:-1]
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2 + self.eps) ** 3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Spiral3(Regularizer):
    def __init__(self, weight: float):
        super(Spiral3, self).__init__()
        self.weight = weight

    def forward(self, factor, time_phase):
        ddiff = factor[1:] - factor[:-1]
        ddiff_pahse = time_phase[1:] - time_phase[:-1]
        rank = int(ddiff.shape[1] / 2)
        rank1 = int(ddiff_pahse.shape[1] / 2)
        diff = (
            torch.sqrt(
                ddiff[:, :rank] ** 2
                + ddiff[:, rank:] ** 2
                + ddiff_pahse[:, :rank1] ** 2
                + ddiff_pahse[:, rank1:] ** 2
            )
            ** 3
        )
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class ER(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
    ):
        super(ER, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}  0.5
        self.scale = scale  # can be adopted from{0.5-1.5}  0.75

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(t**2 + h**2)

        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * norm / h.shape[0]


class ER_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
        eps: float = 1e-5,
    ):
        super(ER_eps, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}  0.5
        self.scale = scale  # can be adopted from{0.5-1.5}  0.75
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(t**2 + h**2)

        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * (norm + self.eps) / h.shape[0]


class ER1(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
    ):
        super(ER1, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}
        self.scale = scale  # can be adopted from{0.5-1.5}

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(t**3 + h**3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * norm / h.shape[0]


class ER1_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
        eps: float = 1e-5,
    ):
        super(ER1_eps, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}
        self.scale = scale  # can be adopted from{0.5-1.5}
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(t**3 + h**3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * (norm + self.eps) / h.shape[0]


class ER2(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
    ):
        super(ER2, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}
        self.scale = scale  # can be adopted from{0.5-1.5}

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(torch.abs(t) ** 3 + torch.abs(h) ** 3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * norm / h.shape[0]


class ER2_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 0.455,
        scale: float = 0.6,
        eps: float = 1e-5,
    ):
        super(ER2_eps, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted
        self.rate = rate  # can be adopted from{0.5-1.5}
        self.scale = scale  # can be adopted from{0.5-1.5}
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(torch.abs(t) ** 3 + torch.abs(h) ** 3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                + self.b * t**2 * r**2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * h**2 * r**2
                + self.a * 2 * h * r * t * r
                + self.a * t**2 * r**2
                + self.b * h**2 * r**2
                - self.b * 2 * h * r * t * r
                + self.b * t**2 * r**2
            )

        return self.weight * (norm + self.eps) / h.shape[0]


class ER_RESCAL(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 1,
        scale: float = 0.5,
    ):
        super(ER_RESCAL, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted. e.g., a=1 b=1.02
        self.rate = rate  # can be adopted from{0-1.5}  1  1.05
        self.scale = scale  # can be adopted from{0-1.5}  0.5  0.455

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(h**2 + t**2)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + 2
                * self.a
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                - 2
                * self.b
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )

        return self.weight * norm / h.shape[0]


class ER_RESCAL_eps(Regularizer):
    # Refs: https://github.com/Lion-ZS/ER
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1,
        rate: float = 1,
        scale: float = 0.5,
        eps: float = 1e-5,
    ):
        super(ER_RESCAL_eps, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted. e.g., a=1 b=1.02
        self.rate = rate  # can be adopted from{0-1.5}  1  1.05
        self.scale = scale  # can be adopted from{0-1.5}  0.5  0.455
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += self.rate * torch.sum(h**2 + t**2)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + 2
                * self.a
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                - 2
                * self.b
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )

        return self.weight * (norm + self.eps) / h.shape[0]


class ER_RESCAL1(Regularizer):
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1.02,
        rate: float = 1,
        scale: float = 0.5,
    ):
        super(ER_RESCAL1, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted from {0-2}. e.g., a=1 b=1.02
        self.rate = rate  # can be adopted from{0-1.5}  1
        self.scale = scale  # can be adopted from{0-1.5}  0.5

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += torch.sum(h**3 + t**3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + 2
                * self.a
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                - 2
                * self.b
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )

        return self.weight * norm / h.shape[0]


class ER_RESCAL1_eps(Regularizer):
    def __init__(
        self,
        weight: float,
        a: float = 1,
        b: float = 1.02,
        rate: float = 1,
        scale: float = 0.5,
        eps: float = 1e-5,
    ):
        super(ER_RESCAL1_eps, self).__init__()
        self.weight = weight
        self.a = a
        self.b = b  # can be adjusted from {0-2}. e.g., a=1 b=1.02
        self.rate = rate  # can be adopted from{0-1.5}  1
        self.scale = scale  # can be adopted from{0-1.5}  0.5
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += torch.sum(h**3 + t**3)
        if self.a == self.b:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )
        else:
            norm += self.scale * torch.sum(
                self.a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                + 2
                * self.a
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.a * torch.bmm(r, t.unsqueeze(-1)) ** 2
                + self.b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
                - 2
                * self.b
                * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1))
                * torch.bmm(r, t.unsqueeze(-1))
                + self.b * torch.bmm(r, t.unsqueeze(-1)) ** 2
            )

        return self.weight * (norm + self.eps) / h.shape[0]


class Fro(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            norm += self.weight * torch.sum(torch.norm(factor, 2) ** 2)

        return norm / factors[0][0].shape[0]


class Fro_eps(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(Fro_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        for factor in factors:
            norm += self.weight * torch.sum(torch.norm(factor, 2) ** 2 + self.eps)

        return norm / factors[0][0].shape[0]


class DURA(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        h, r, t, tau = factors

        norm += torch.sum(t**2 + h**2)
        norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]


class DURA_eps(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(DURA_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0

        h, r, t = factors

        norm += torch.sum(t**2 + h**2)
        norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * (norm + self.eps) / h.shape[0]


class DURA_RESCAL(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        h, r, t = factors
        norm += torch.sum(h**2 + t**2)
        norm += torch.sum(
            torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
            + torch.bmm(r, t.unsqueeze(-1)) ** 2
        )

        return self.weight * norm / h.shape[0]


class DURA_RESCAL_eps(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(DURA_RESCAL_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        h, r, t = factors
        norm += torch.sum(h**2 + t**2)
        norm += torch.sum(
            torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
            + torch.bmm(r, t.unsqueeze(-1)) ** 2
        )

        return self.weight * (norm + self.eps) / h.shape[0]


class DURA_RESCAL_W(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float):
        super(DURA_RESCAL_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        h, r, t = factors
        norm += 2.0 * torch.sum(h**2 + t**2)
        norm += 0.5 * torch.sum(
            torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
            + torch.bmm(r, t.unsqueeze(-1)) ** 2
        )

        return self.weight * norm / h.shape[0]


class DURA_RESCAL_W_eps(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(DURA_RESCAL_W_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        h, r, t = factors
        norm += 2.0 * torch.sum(h**2 + t**2)
        norm += 0.5 * torch.sum(
            torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2
            + torch.bmm(r, t.unsqueeze(-1)) ** 2
        )

        return self.weight * (norm + self.eps) / h.shape[0]


class DURA_W(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        h, r, t = factors

        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]


class DURA_W_eps(Regularizer):
    # Refs: https://github.com/MIRALab-USTC/KGE-DURA/blob/main/code/regularizers.py
    def __init__(self, weight: float, eps: float = 1e-5):
        super(DURA_W_eps, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, factors):
        norm = 0
        h, r, t = factors

        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * (norm + self.eps) / h.shape[0]


class Temporal_MSE_Regularizer(Regularizer):
    def __init__(self, weight: float):
        super(Temporal_MSE_Regularizer, self).__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor], time=None):
        ddiff = factor[1:] - factor[:-1]
        diff = ddiff**2
        time_diff = torch.sum(diff) / (factor.shape[0] - 1)
        return self.weight * time_diff


class Temporal_Gaussian_Regularizer(Regularizer):
    def __init__(self, weight: float):
        super(Temporal_Gaussian_Regularizer, self).__init__()
        self.weight = weight
        self.sigma = 0.01

    def forward(self, factor: Tuple[torch.Tensor], time=None):
        ddiff = factor[1:] - factor[:-1]
        diff = -(ddiff**2) / (2 * self.sigma * self.sigma)
        time_diff = self.weight * torch.exp(torch.sum(diff) / (factor.shape[0] - 1))
        return -self.weight * time_diff


class Temporal_Cosine_Regularizer(Regularizer):
    def __init__(self, weight):
        super(Temporal_Cosine_Regularizer, self).__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor], time=None):
        factor = nn.functional.normalize(factor)
        sim = torch.mm(factor, factor.T)
        N = factor.shape[0]
        positive1 = torch.sum(torch.diag(sim, 1)) / (N - 1)
        positive2 = torch.sum(torch.diag(sim, -1)) / (N - 1)
        time_diff = positive1 + positive2
        return -self.weight * time_diff


class Temporal_CL_Regularizer(Regularizer):
    def __init__(self, weight) -> None:
        super(Temporal_CL_Regularizer, self).__init__()
        self.weight = weight

    def forward(self, factor: Tuple[torch.Tensor], time=None):
        # contrastive learning
        criterion_node = nn.CrossEntropyLoss(reduction="sum")
        factor = nn.functional.normalize(factor)
        sim = torch.mm(factor, factor.T)
        N = factor.shape[0]
        labels_node = torch.zeros(N - 1).to(factor.device).long()

        positive1 = torch.diag(sim, 1)
        positive_samples = positive1.reshape(N - 1, 1)

        ################
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N - 1):
            mask[i, 1 + i] = 0
        for i in range(N - 1):
            mask[i + 1, i] = 0
        mask = mask.bool()
        negative_samples = sim[mask].reshape(N - 1, N - 2)
        logits_node = torch.cat((positive_samples, negative_samples), dim=1)
        loss_t = criterion_node(logits_node, labels_node)
        loss_t /= N
        return -self.weight * loss_t
