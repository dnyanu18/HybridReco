import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')

    # =========================
    # Training
    # =========================
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch', default=2048, type=int, help='train batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='test batch size')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')

    # =========================
    # Model
    # =========================
    parser.add_argument('--latdim', default=64, type=int, help='embedding dimension')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of GCN layers')

    # =========================
    # Evaluation
    # =========================
    parser.add_argument('--topk', default=20, type=int, help='top-K recommendation')

    # =========================
    # Dataset
    # =========================
    parser.add_argument('--data', default='lastfm', type=str)

    # =========================
    # 🔥 Self-Supervised Learning
    # =========================
    parser.add_argument('--ssl_reg', default=0.05, type=float, help='contrastive loss weight')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature')

    # =========================
    # Training Control
    # =========================
    parser.add_argument('--tstEpoch', default=1, type=int, help='test every N epochs')
    parser.add_argument('--seed', default=421, type=int, help='random seed')

    # =========================
    # 🔥 Extra Features
    # =========================
    parser.add_argument('--early_stop', default=5, type=int, help='early stopping patience')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')

    # =========================
    # 🔥 IMPORTANT FIX (FastAPI / Uvicorn compatibility)
    # =========================
    args, unknown = parser.parse_known_args()

    return args


# Global args
args = ParseArgs()