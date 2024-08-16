import os
import logging

import torch

from models import (
    ComplEx,
    TComplEx,
    TNTComplEx,
    TPComplEx,
    MPComplEx,
    nMPComplEx, 
    MPComplExCOPS
)
from utils.arguments import parse_args
from utils.datasets import TemporalDataset
from utils.general import Path, set_logger, increment_path, avg_both
from utils.optimizers import TKBCOptimizer, IKBCOptimizer
from utils.regularizers import pick_embregularizer, pick_timeregularizer
from utils.schedulers import pick_scheduler
from utils.torch_optimizers import pick_optimizer
from utils.torch_utils import init_seeds


if __name__ == "__main__":
    args = parse_args()

    curr_dir = Path(__file__).resolve().parent
    if args.dataset != "ICEWS05-15":
        dataname = args.dataset
    elif args.dataset == "ICEWS05-15":
        dataname = "ICEWS0515"
    else:
        dataname = "Unkown"
    name = Path(
        args.save_path + os.sep + args.model + "_" + dataname + "_" + args.model_id
    )

    if not os.path.exists(curr_dir / name):
        os.makedirs(curr_dir / name)
        args.save_path = curr_dir / name
    else:
        args.save_path = str(
            increment_path(curr_dir / name, exist_ok=args.exist_ok, mkdir=True)
        )

    set_logger(args)
    logging.info(args)

    init_seeds(args.random_seed)
    logging.info(f"Random seed set as {args.random_seed}")

    dataset = TemporalDataset(args.dataset)

    sizes = dataset.get_shape()

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    if args.model == "ComplEx":
        model = ComplEx(sizes, args.rank)
    elif args.model == "TComplEx":
        model = TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb)
    elif args.model == "TNTComplEx":
        model = TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb)
    elif args.model == "TPComplEx":
        model = TPComplEx(sizes, args.rank, no_time_emb=args.no_time_emb)
    elif args.model == "MPComplEx":
        model = MPComplEx(sizes, args.rank, no_time_emb=args.no_time_emb)
    elif args.model == "nMPComplEx":
        model = nMPComplEx(sizes, args.rank, no_time_emb=args.no_time_emb)
    elif args.model == "MPComplEx_OPS":
        model = MPComplExCOPS(sizes, args.rank, no_time_emb=args.no_time_emb)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    if args.cuda:
        model = model.cuda()

    opt = pick_optimizer(model, args.optim, args.learning_rate,
                         args.beta1, args.weight_decay)
    scheduler = pick_scheduler(opt, args.scheduler)

    emb_reg = pick_embregularizer(args.emb_reg_type, args.emb_reg)
    time_reg = pick_timeregularizer(args.time_reg_type, args.time_reg)

    if args.do_test:
        model.load_state_dict(torch.load(args.check_point))

        if args.dataset == "ICEWS05-15":
            valid, test, train, T1 = [
                avg_both(*dataset.eval(model, split, -
                                       1 if split != "train" else 50000))
                for split in ["valid", "test", "train", "test_Arrest"]
            ]

            logging.info(
                "valid: MRR:={}; hits@[1,3,10]={}".format(
                    valid["MRR"], valid["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "test: MRR={}; hits@[1,3,10]={}".format(
                    test["MRR"], test["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "train: MRR={}; hits@[1,3,10]={}".format(
                    train["MRR"], train["hits@[1,3,10]"].tolist()
                )
            )

            logging.info(
                "T1: MRR={}; hits@[1,3,10]={}".format(
                    T1["MRR"], T1["hits@[1,3,10]"].tolist()
                )
            )

        else:

            valid, test, train, syms, invs, comps, ct_invs, ct_hierarchy, ct_intersection, ct_comps, aggregation, associativity = [
                avg_both(*dataset.eval(model, split, -
                                       1 if split != "train" else 50000))
                for split in ["valid", "test", "train", "syms", "invs", "comps", "ct_invs", "ct_hierarchy", "ct_intersection", "ct_comps", "aggregation", "associativity"]
                # for split in ["valid", "test", "train", "syms", "invs", "comps", "ct_invs", "ct_hierarchy", "ct_intersection", ]
                # for split in ["valid", "test", "train", "syms", "invs", "comps", "ct_invs", "ct_hierarchy", ]
                # for split in ["valid", "syms", "train"]
                # for split in ["valid", , "train"]
                # for split in ["valid", , "train"]
            ]
            logging.info(
                "valid: MRR:={}; hits@[1,3,10]={}".format(
                    valid["MRR"], valid["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "test: MRR={}; hits@[1,3,10]={}".format(
                    test["MRR"], test["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "train: MRR={}; hits@[1,3,10]={}".format(
                    train["MRR"], train["hits@[1,3,10]"].tolist()
                )
            )

            logging.info(
                "syms: MRR={}; hits@[1,3,10]={}".format(
                    syms["MRR"], syms["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "invs: MRR={}; hits@[1,3,10]={}".format(
                    invs["MRR"], invs["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "comps: MRR={}; hits@[1,3,10]={}".format(
                    comps["MRR"], comps["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "ct_invs: MRR={}; hits@[1,3,10]={}".format(
                    ct_invs["MRR"], ct_invs["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "ct_hierarchy: MRR={}; hits@[1,3,10]={}".format(
                    ct_hierarchy["MRR"], ct_hierarchy["hits@[1,3,10]"].tolist()
                )
            )

            logging.info(
                "ct_intersection: MRR={}; hits@[1,3,10]={}".format(
                    ct_intersection["MRR"], ct_intersection["hits@[1,3,10]"].tolist()
                )
            )
            logging.info(
                "ct_comps: MRR={}; hits@[1,3,10]={}".format(
                    ct_comps["MRR"], ct_comps["hits@[1,3,10]"].tolist()
                )
            )

            logging.info(
                "aggregation: MRR={}; hits@[1,3,10]={}".format(
                    aggregation["MRR"], aggregation["hits@[1,3,10]"].tolist()
                )
            )

            logging.info(
                "associativity: MRR={}; hits@[1,3,10]={}".format(
                    associativity["MRR"], associativity["hits@[1,3,10]"].tolist()
                )
            )

    else:
        patience = 0
        mrr_std = 0

        curve = {"train": [], "valid": [], "test": []}

        logging.info("Start Training......")
        for epoch in range(args.max_epochs):
            logging.info("Epoch: {}".format(epoch))

            examples = torch.from_numpy(dataset.get_train().astype("int64"))

            model.train()
            if dataset.has_intervals():
                logging.info("Interval Training......")
                optimizer = IKBCOptimizer(
                    model,
                    emb_reg,
                    time_reg,
                    opt,
                    scheduler,
                    dataset,
                    batch_size=args.batch_size,
                    grad_norm=args.grad_norm,
                    label_smoothing=args.label_smoothing,
                )
                tl_fit, tl_reg, tl_time, tl = optimizer.epoch(examples)

            else:
                logging.info("Time Training......")
                optimizer = TKBCOptimizer(
                    model,
                    emb_reg,
                    time_reg,
                    opt,
                    scheduler,
                    batch_size=args.batch_size,
                    grad_norm=args.grad_norm,
                    label_smoothing=args.label_smoothing,
                )
                tl_fit, tl_reg, tl_time, tl = optimizer.epoch(examples)

            logging.info(
                "TL Fit = {}; TL Reg = {}; TL time = {}; TL = {}".format(
                    tl_fit, tl_reg, tl_time, tl
                )
            )

            if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
                if dataset.has_intervals():
                    valid, test, train = [
                        dataset.eval(model, split, -1 if split !=
                                     "train" else 50000)
                        for split in ["valid", "test", "train"]
                    ]
                    logging.info(
                        "valid: MRR:={}; hits@[1,3,10]={}".format(
                            valid["MRR"], valid["hits@[1,3,10]"].tolist()
                        )
                    )
                    logging.info(
                        "test: MRR={}; hits@[1,3,10]={}".format(
                            test["MRR"], test["hits@[1,3,10]"].tolist()
                        )
                    )
                    logging.info(
                        "train: MRR={}; hits@[1,3,10]={}".format(
                            train["MRR"], train["hits@[1,3,10]"].tolist()
                        )
                    )

                else:
                    valid, test, train = [
                        avg_both(
                            *dataset.eval(
                                model, split, -1 if split != "train" else 50000
                            )
                        )
                        for split in ["valid", "test", "train"]
                    ]
                    logging.info(
                        "valid: MRR:={}; hits@[1,3,10]={}".format(
                            valid["MRR"], valid["hits@[1,3,10]"].tolist()
                        )
                    )
                    logging.info(
                        "test: MRR={}; hits@[1,3,10]={}".format(
                            test["MRR"], test["hits@[1,3,10]"].tolist()
                        )
                    )
                    logging.info(
                        "train: MRR={}; hits@[1,3,10]={}".format(
                            train["MRR"], train["hits@[1,3,10]"].tolist()
                        )
                    )

                if args.do_save:
                    torch.save(model, args.save_path)

                # early-stop with patience
                mrr_valid = valid["MRR"]
                if mrr_valid < mrr_std:
                    patience += 1
                    if patience >= 5:
                        logging.info("Early stopping ...")
                        break
                else:
                    patience = 0
                    mrr_std = mrr_valid
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.save_path, args.model + ".pkl"),
                    )

                curve["valid"].append(valid)
                if not dataset.has_intervals():
                    curve["train"].append(train)
                    logging.info("TRAIN: {}".format(train))

                logging.info("VALID: {}".format(valid))

        logging.info("Start Testing......")
        model.load_state_dict(
            torch.load(os.path.join(args.save_path, args.model + ".pkl"))
        )
        results = avg_both(*dataset.eval(model, "test", -1))
        logging.info("TEST: {}".format(results))
        # os.remove(os.path.join(args.save_path, args.model + ".pkl"))
