import tqdm
import torch
from torch import nn
from torch import optim

from models.TKBCModel import TKBCModel
from utils.regularizers import Regularizer
from utils.datasets import TemporalDataset

from utils.loss import LabelSmoothCrossEntropyLoss


class TKBCOptimizer(object):
    def __init__(
        self,
        model: TKBCModel,
        emb_regularizer: Regularizer,
        temporal_regularizer: Regularizer,
        optimizer: optim.Optimizer,
        scheduler,
        batch_size: int = 256,
        verbose: bool = True,
        grad_norm: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.verbose = verbose
        self.grad_norm = grad_norm
        self.label_smoothing = label_smoothing

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.label_smoothing)

        tl_fit = 0
        tl_reg = 0
        tl_time = 0
        tl = 0

        with tqdm.tqdm(
            total=examples.shape[0], unit="ex", disable=not self.verbose
        ) as bar:
            bar.set_description(f"train loss")
            b_begin = 0
            time_phase = None
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin : b_begin + self.batch_size
                ].cuda()

                results = self.model.forward(input_batch)

                if len(results) == 4:
                    predictions, factors, time, time_phase = results
                elif len(results) == 3:
                    predictions, factors, time = results

                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)               

                # regularization
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    if time_phase is not None:
                        l_time = self.temporal_regularizer.forward(time, time_phase)
                    else:
                        l_time = self.temporal_regularizer.forward(time)

                # Refs: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/10
                # l_reg = torch.where(torch.isnan(l_reg), torch.zeros_like(l_reg), l_reg)
                # l_time = torch.where(
                #     torch.isnan(l_time), torch.zeros_like(l_time), l_time
                # )
                
                l = l_fit + l_reg + l_time

                tl_fit += l_fit
                tl_reg += l_reg
                tl_time += l_time
                tl += l

                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_norm
                )

                self.optimizer.step()
                self.optimizer.zero_grad()

                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f"{l_fit.item():.0f}",
                    reg=f"{l_reg.item():.0f}",
                    cont=f"{l_time.item():.0f}",
                )

        if self.scheduler != None:
            self.scheduler.step()
            
        tl_fit /= examples.shape[0]
        tl_reg /= examples.shape[0]
        tl_time /= examples.shape[0]
        tl /= examples.shape[0]

        return tl_fit, tl_reg, tl_time, tl


class IKBCOptimizer(object):
    def __init__(
        self,
        model: TKBCModel,
        emb_regularizer: Regularizer,
        temporal_regularizer: Regularizer,
        optimizer: optim.Optimizer,
        scheduler,
        dataset: TemporalDataset,
        batch_size: int = 256,
        verbose: bool = True,
        grad_norm: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.dataset = dataset
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.verbose = verbose
        self.grad_norm = grad_norm
        self.label_smoothing = label_smoothing

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=self.label_smoothing)

        tl_fit = 0
        tl_reg = 0
        tl_time = 0
        tl = 0

        with tqdm.tqdm(
            total=examples.shape[0], unit="ex", disable=not self.verbose
        ) as bar:
            bar.set_description(f"train loss")
            b_begin = 0
            while b_begin < examples.shape[0]:
                time_range = actual_examples[b_begin : b_begin + self.batch_size].cuda()

                ## RHS Prediction loss
                sampled_time = (
                    (
                        torch.rand(time_range.shape[0]).cuda()
                        * (time_range[:, 4] - time_range[:, 3]).float()
                        + time_range[:, 3].float()
                    )
                    .round()
                    .long()
                )
                with_time = torch.cat(
                    (time_range[:, 0:3], sampled_time.unsqueeze(1)), 1
                )

                results = self.model.forward(with_time)

                if len(results) == 4:
                    predictions, factors, time, time_phase = results
                elif len(results) == 3:
                    predictions, factors, time = results

                # predictions, factors, time = self.model.forward(with_time)
                truth = with_time[:, 2]

                l_fit = loss(predictions, truth)

                ## Time prediction loss (ie cross entropy over time)
                time_loss = 0.0
                if self.model.has_time():
                    filtering = ~(
                        (time_range[:, 3] == 0)
                        * (time_range[:, 4] == (self.dataset.n_timestamps - 1))
                    )  # NOT no begin and no end
                    these_examples = time_range[filtering, :]
                    truth = (
                        (
                            torch.rand(these_examples.shape[0]).cuda()
                            * (these_examples[:, 4] - these_examples[:, 3]).float()
                            + these_examples[:, 3].float()
                        )
                        .round()
                        .long()
                    )
                    time_predictions = self.model.forward_over_time(
                        these_examples[:, :3].cuda().long()
                    )
                    time_loss = loss(time_predictions, truth.cuda())

                # regularization
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    if time_phase is not None:
                        l_time = self.temporal_regularizer.forward(time, time_phase)
                    else:
                        l_time = self.temporal_regularizer.forward(time)

                # Refs: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/10
                # l_reg = torch.where(torch.isnan(l_reg), torch.zeros_like(l_reg), l_reg)
                # l_time = torch.where(
                #     torch.isnan(l_time), torch.zeros_like(l_time), l_time
                # )

                l = l_fit + l_reg + l_time

                tl_fit += l_fit
                tl_reg += l_reg
                tl_time += l_time
                tl += l

                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                b_begin += self.batch_size
                bar.update(with_time.shape[0])
                bar.set_postfix(
                    loss=f"{l_fit.item():.0f}",
                    loss_time=f"{time_loss if type(time_loss) == float else time_loss.item() :.0f}",
                    reg=f"{l_reg.item():.0f}",
                    cont=f"{l_time.item():.4f}",
                )

        if self.scheduler != None:
            self.scheduler.step()
        tl_fit /= examples.shape[0]
        tl_reg /= examples.shape[0]
        tl_time /= examples.shape[0]
        tl /= examples.shape[0]

        return tl_fit, tl_reg, tl_time, tl
