from typing import Tuple, List

import torch
from torch import nn

from models.TKBCModel import TKBCModel

from utils.activations import Acts


class nMPComplEx(TKBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int, int],
        rank: int,
        no_time_emb=False,
        init_size: float = 1e-2,
        acts: List[str] = ["Linear", "Linear", "Linear"],
        coefficients: List[float] = [2, 1, 2, 1, 0.75],
    ):

        # [2, 1.5, 2, 1.5, 0.75]
        # [2, 1.5, 2, 1.5, 0.5]
        # [2.5, 2, 2.5, 2, 0.75]
        # [2.5, 2, 2.5, 2, 0.5]

        # Linear Linear Linear
        # PReLU PReLU PReLU
        # GELU GELU GELU
        # Tanh Tanh Tanh

        super(nMPComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.alpha1 = float(coefficients[0])
        self.alpha2 = float(coefficients[1])

        self.beta1 = float(coefficients[2])
        self.beta2 = float(coefficients[3])

        self.gamma = float(coefficients[4])

        self.f, self.g, self.h = Acts(acts[0]), Acts(acts[1]), Acts(acts[2])
        self.f_rel = Acts("Linear")

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(sizes[0], 2 * rank, sparse=True),
                nn.Embedding(sizes[1], 2 * rank, sparse=True),
                nn.Embedding(sizes[2], 6 * rank, sparse=True),
            ]
        )
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def nl_residual(self, x, act_func, residual=True):
        if residual:
            return act_func(x) + x
        else:
            return act_func(x)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        rcontext_rel = rel[:, : self.rank] * time[:, : self.rank]
        icontext_rel = (
            rel[:, self.rank : 2 * self.rank] * time[:, self.rank : 2 * self.rank]
        )

        # Baseline
        # lhs = (
        #     lhs[:, : self.rank] + time[:, 2 * self.rank : 3 * self.rank],
        #     lhs[:, self.rank :] + time[:, 3 * self.rank : 4 * self.rank],
        # )

        # Improve solution 1
        lhs = (
            self.nl_residual(
                self.alpha1 * lhs[:, : self.rank]
                + self.alpha2 * time[:, 2 * self.rank : 3 * self.rank],
                self.f,
            ),
            self.nl_residual(
                self.alpha1 * lhs[:, self.rank :]
                + self.alpha2 * time[:, 3 * self.rank : 4 * self.rank],
                self.f,
            ),
        )

        # Baseline
        # rhs = (
        #     rhs[:, : self.rank] + time[:, 4 * self.rank : 5 * self.rank],
        #     rhs[:, self.rank :] + time[:, 5 * self.rank : 6 * self.rank],
        # )

        # Improve solution 1
        rhs = (
            self.nl_residual(
                self.beta1 * rhs[:, : self.rank]
                + self.beta2 * time[:, 4 * self.rank : 5 * self.rank],
                self.g,
            ),
            self.nl_residual(
                self.beta1 * rhs[:, self.rank :]
                + self.beta2 * time[:, 5 * self.rank : 6 * self.rank],
                self.g,
            ),
        )

        # rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]
        rel = (
            self.f_rel(
                self.gamma * rel[:, : self.rank] + (1 - self.gamma) * rcontext_rel
            ),
            self.f_rel(
                self.gamma * rel[:, self.rank : 2 * self.rank]
                + (1 - self.gamma) * icontext_rel
            ),
        )

        # time = time[:, : self.rank], time[:, self.rank : 2 * self.rank]
        time = self.nl_residual(time[:, : self.rank], self.h), self.nl_residual(
            time[:, self.rank : 2 * self.rank], self.h
        )

        return torch.sum(
            (
                lhs[0] * rel[0] * time[0]
                - lhs[1] * rel[1] * time[0]
                - lhs[1] * rel[0] * time[1]
                - lhs[0] * rel[1] * time[1]
            )
            * rhs[0]
            + (
                lhs[1] * rel[0] * time[0]
                + lhs[0] * rel[1] * time[0]
                + lhs[0] * rel[0] * time[1]
                - lhs[1] * rel[1] * time[1]
            )
            * rhs[1],
            1,
            keepdim=True,
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        rcontext_rel = rel[:, : self.rank] * time[:, : self.rank]
        icontext_rel = (
            rel[:, self.rank : 2 * self.rank] * time[:, self.rank : 2 * self.rank]
        )

        # Baseline
        # lhs = (
        #     lhs[:, : self.rank] + time[:, 2 * self.rank : 3 * self.rank],
        #     lhs[:, self.rank :] + time[:, 3 * self.rank : 4 * self.rank],
        # )

        # Improve solution 1
        lhs = (
            self.nl_residual(
                self.alpha1 * lhs[:, : self.rank]
                + self.alpha2 * time[:, 2 * self.rank : 3 * self.rank],
                self.f,
            ),
            self.nl_residual(
                self.alpha1 * lhs[:, self.rank :]
                + self.alpha2 * time[:, 3 * self.rank : 4 * self.rank],
                self.f,
            ),
        )

        # Baseline
        # rhs = (
        #     rhs[:, : self.rank] + time[:, 4 * self.rank : 5 * self.rank],
        #     rhs[:, self.rank :] + time[:, 5 * self.rank : 6 * self.rank],
        # )

        # Improve solution 1
        rhs = (
            self.nl_residual(
                self.beta1 * rhs[:, : self.rank]
                + self.beta2 * time[:, 4 * self.rank : 5 * self.rank],
                self.g,
            ),
            self.nl_residual(
                self.beta1 * rhs[:, self.rank :]
                + self.beta2 * time[:, 5 * self.rank : 6 * self.rank],
                self.g,
            ),
        )

        bias_t_r = time[:, 4 * self.rank : 5 * self.rank]
        bias_t_i = time[:, 5 * self.rank : 6 * self.rank]

        # time = time[:, : self.rank], time[:, self.rank : 2 * self.rank]
        time = self.nl_residual(time[:, : self.rank], self.h), self.nl_residual(
            time[:, self.rank : 2 * self.rank], self.h
        )

        right = self.embeddings[0].weight
        right = right[:, : self.rank], right[:, self.rank :]

        # rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]
        rel = (
            self.f_rel(
                self.gamma * rel[:, : self.rank] + (1 - self.gamma) * rcontext_rel
            ),
            self.f_rel(
                self.gamma * rel[:, self.rank : 2 * self.rank]
                + (1 - self.gamma) * icontext_rel
            ),
        )

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
            (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t()
                + torch.sum(
                    (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r,
                    1,
                    keepdim=True,
                )
                + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
                + torch.sum(
                    (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i,
                    1,
                    keepdim=True,
                )
            ),
            (
                torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
            ),
            (
                self.embeddings[2].weight[:-1]
                if self.no_time_emb
                else self.embeddings[2].weight
            ),
        )

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return (
            self.embeddings[0]
            .weight.data[chunk_begin : chunk_begin + chunk_size]
            .transpose(0, 1)
        )

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        rcontext_rel = rel[:, : self.rank] * time[:, : self.rank]
        icontext_rel = (
            rel[:, self.rank : 2 * self.rank] * time[:, self.rank : 2 * self.rank]
        )

        # Baseline
        # lhs = (
        #     lhs[:, : self.rank] + time[:, 2 * self.rank : 3 * self.rank],
        #     lhs[:, self.rank :] + time[:, 3 * self.rank : 4 * self.rank],
        # )

        # Improve solution 1
        lhs = (
            self.nl_residual(
                self.alpha1 * lhs[:, : self.rank]
                + self.alpha2 * time[:, 2 * self.rank : 3 * self.rank],
                self.f,
            ),
            self.nl_residual(
                self.alpha1 * lhs[:, self.rank :]
                + self.alpha2 * time[:, 3 * self.rank : 4 * self.rank],
                self.f,
            ),
        )

        # rel = rel[:, : self.rank], rel[:, self.rank : 2 * self.rank]
        rel = (
            self.f_rel(
                self.gamma * rel[:, : self.rank] + (1 - self.gamma) * rcontext_rel
            ),
            self.f_rel(
                self.gamma * rel[:, self.rank : 2 * self.rank]
                + (1 - self.gamma) * icontext_rel
            ),
        )

        # time = time[:, : self.rank], time[:, self.rank : 2 * self.rank]
        time = self.nl_residual(time[:, : self.rank], self.h), self.nl_residual(
            time[:, self.rank : 2 * self.rank], self.h
        )

        return torch.cat(
            [
                lhs[0] * rel[0] * time[0]
                - lhs[1] * rel[1] * time[0]
                - lhs[1] * rel[0] * time[1]
                - lhs[0] * rel[1] * time[1],
                lhs[1] * rel[0] * time[0]
                + lhs[0] * rel[1] * time[0]
                + lhs[0] * rel[0] * time[1]
                - lhs[1] * rel[1] * time[1],
            ],
            1,
        )
