import torch


def pick_scheduler(opt, scheduler: str):
    if scheduler == "LambdaLR":
        # LAMBDA LR (0.93)
        lambda1 = lambda epoch: 0.65**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    elif scheduler == "MultiplicativeLR":
        # MultiplicativeLR
        lmbda = lambda epoch: 0.65**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lmbda)
    elif scheduler == "StepLR":
        # StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    elif scheduler == "MultiStepLR":
        # MultiStepLR (0.95)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[6, 8, 9], gamma=0.1
        )
    elif scheduler == "ExponentialLR":
        # ExponentialLR
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1)
    elif scheduler == "CosineAnnealingLR":
        # CosineAnnealingLR (0.97)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0)
    elif scheduler == "CyclicLR_triangular":
        # CyclicLR - triangular
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular"
        )
    elif scheduler == "CyclicLR_triangular2":
        # CyclicLR - triangular2
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular2"
        )
    elif scheduler == "CyclicLR_exp_range":
        # CyclicLR - exp_range
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85
        )
    elif scheduler == "OneCycleLR_cos":
        # OneCycleLR - cos
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, steps_per_epoch=10, epochs=10
        )
    elif scheduler == "OneCycleLR_linear":
        # OneCycleLR - linear
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, steps_per_epoch=10, epochs=10, anneal_strategy="linear"
        )
    elif scheduler == "CosineAnnealingWarmRestarts":
        # CosineAnnealingWarmRestarts (0.97)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1
        )
    elif scheduler == "None":
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler {scheduler} not implemented.")
    
    return scheduler
