import argparse


def parse_args(args=None):
    """
    Function for parsing argument.

    Args:
        args (_type_, optional): _description_. Defaults to None.
    """
    parser = argparse.ArgumentParser(
        description="Training and Testing Temporal Knowledge Graph Completion Models",
        usage="learner.py [<args>] [-h | --help]",
    )

    parser.add_argument("--dataset", default="ICEWS14", type=str, help="Dataset name")
    models = [
        "ComplEx",
        "TComplEx",
        "TNTComplEx",
        "TPComplEx",
        "MPComplEx",
        "MPComplEx_COPS",
        "nMPComplEx",
    ]
    parser.add_argument(
        "--random-seed", default=2024, type=int, help="init random seed"
    )

    parser.add_argument(
        "--model",
        default="TPComplEx",
        choices=models,
        help="Model in {}".format(models),
    )
    parser.add_argument(
        "--cuda", action="store_true", help="whether to use GPU or not."
    )
    parser.add_argument(
        "--save-path",
        default="./runs/",
        type=str,
        help="trained model checkpoint path.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment.",
    )

    parser.add_argument("--max-epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument(
        "--valid-freq", default=5, type=int, help="Number of epochs between each valid."
    )
    parser.add_argument("--rank", default=100, type=int, help="Factorization rank.")
    parser.add_argument("--batch-size", default=1000, type=int, help="Batch size.")
    parser.add_argument(
        "--learning-rate", "-lr", default=1e-1, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--emb-reg", default=0.0, type=float, help="Embedding regularizer strength"
    )
    parser.add_argument(
        "--emb-reg-type", default="N3", type=str, help="Type of embedding regularizer"
    )
    parser.add_argument(
        "--time-reg", default=0.0, type=float, help="Timestamp regularizer strength"
    )
    parser.add_argument(
        "--time-reg-type",
        default="Lambda3",
        type=str,
        help="Type of timestamp regularizer",
    )
    parser.add_argument(
        "--no-time-emb",
        default=False,
        action="store_true",
        help="Use a specific embedding for non temporal relations",
    )
    parser.add_argument("--cycle", default=365, type=int, help="time range for sharing")
    parser.add_argument(
        "--optim",
        default="Adagrad",
        type=str,
        help="Optimizer for training TKC models.",
    )
    parser.add_argument(
        "--beta1",
        default=0.937,
        type=float,
        help="Beta1 for Lion Optimizer in training TKC models.",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0005,
        type=float,
        help="Weight decay for Lion Optimizer in training TKC models.",
    )
    parser.add_argument(
        "--eps",
        default=1e-10,
        type=float,
        help="term added to the denominator to improve numerical stability (default: 1e-10).",
    )
    parser.add_argument(
        "--grad-norm",
        type=float,
        default=1.0,
        help="norm to clip gradient to",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="label smoothing rate.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealingLR",
        help="Scheduler.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="operator for tổng hợp",
        help="Scheduler.",
    )
    parser.add_argument(
        "-test", "--do-test", action="store_true"
    )  # action='store_true'
    parser.add_argument("-save", "--do_save", action="store_true")
    parser.add_argument("-id", "--model_id", type=str, default="0")
    parser.add_argument(
        "--check-point",
        type=str,
        default="check point",
        help="check point.",
    )
    parser.add_argument(
        "--acts",
        nargs="+",
        help="Activation functions for NTPComplEx",
        default=["Tanh", "Tanh", "Tanh"],
    )
    parser.add_argument("--coeffs", nargs="+", help="coeffs", default=[1, 1, 1, 1, 1])
    return parser.parse_args(args)
