from typing import Tuple

import torch
from torch import nn

from models.TKBCModel import TKBCModel


class TComplEx(TKBCModel):
    def __init__(
        self,
        sizes: Tuple[int, int, int, int],
        rank: int,
        no_time_emb=False,
        init_size: float = 1e-2,
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(s, 2 * rank, sparse=True)
                for s in [sizes[0], sizes[1], sizes[3]]
            ]
        )
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        rhs = rhs[:, : self.rank], rhs[:, self.rank :]
        time = time[:, : self.rank], time[:, self.rank :]

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

        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        rhs = rhs[:, : self.rank], rhs[:, self.rank :]
        time = time[:, : self.rank], time[:, self.rank :]

        right = self.embeddings[0].weight
        right = right[:, : self.rank], right[:, self.rank :]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        pred = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + (
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ) @ right[1].t()

        regularizer = (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
        )

        weight = (
            self.embeddings[2].weight[:-1]
            if self.no_time_emb
            else self.embeddings[2].weight
        )

        return (pred, regularizer, weight)

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        rhs = rhs[:, : self.rank], rhs[:, self.rank :]
        time = time[:, : self.rank], time[:, self.rank :]

        return (
            lhs[0] * rel[0] * rhs[0]
            - lhs[1] * rel[1] * rhs[0]
            - lhs[1] * rel[0] * rhs[1]
            + lhs[0] * rel[1] * rhs[1]
        ) @ time[0].t() + (
            lhs[1] * rel[0] * rhs[0]
            - lhs[0] * rel[1] * rhs[0]
            + lhs[0] * rel[0] * rhs[1]
            - lhs[1] * rel[1] * rhs[1]
        ) @ time[
            1
        ].t()

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
        lhs = lhs[:, : self.rank], lhs[:, self.rank :]
        rel = rel[:, : self.rank], rel[:, self.rank :]
        time = time[:, : self.rank], time[:, self.rank :]
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
