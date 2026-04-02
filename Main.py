import torch
import numpy as np
from Params import args
from Model import Model
from DataHandler import DataHandler
from Utils.TimeLogger import log
import random
import os


# =========================
# TRAINER
# =========================
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', len(self.handler.trnLoader.dataset))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using Device:", self.device)

        # Model
        self.model = Model().to(self.device)

        # Optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-5
        )

    # =========================
    # TRAIN
    # =========================
    def trainEpoch(self):
        self.model.train()

        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()

        total_loss = 0

        adj = self.handler.torchBiAdj.to(self.device)
        user_seq_map = self.handler.user_seq_map.to(self.device)

        for i, (users, pos, neg) in enumerate(trnLoader):
            users = users.long().to(self.device)
            pos = pos.long().to(self.device)
            neg = neg.long().to(self.device)

            self.opt.zero_grad()

            # ✅ FIXED CALL (REMOVED None)
            loss = self.model.total_loss(users, pos, neg, adj, user_seq_map)

            loss.backward()
            self.opt.step()

            total_loss += loss.item()

            log(f"Step {i}/{len(trnLoader)}: loss={loss.item():.4f}", save=False, oneline=True)

        return total_loss / len(trnLoader)

    # =========================
    # TEST (Recall + NDCG)
    # =========================
    def testEpoch(self):
        self.model.eval()

        adj = self.handler.torchBiAdj.to(self.device)
        user_seq_map = self.handler.user_seq_map.to(self.device)

        tstLoader = self.handler.tstLoader

        epRecall, epNdcg = 0, 0
        num = len(tstLoader.dataset)

        with torch.no_grad():
            # ✅ Correct forward call
            user_emb, item_emb = self.model(adj, user_seq_map)

            for users, trnMask in tstLoader:
                users = users.long().to(self.device)
                trnMask = trnMask.to(self.device)

                scores = torch.matmul(user_emb[users], item_emb.T)

                # Mask training items
                scores = scores * (1 - trnMask) - trnMask * 1e8

                _, topLocs = torch.topk(scores, args.topk)

                recall, ndcg = self.calcRes(
                    topLocs.cpu().numpy(),
                    self.handler.tstLoader.dataset.tstLocs,
                    users.cpu().numpy()
                )

                epRecall += recall
                epNdcg += ndcg

        return {
            'Recall': epRecall / num,
            'NDCG': epNdcg / num
        }

    # =========================
    # METRICS
    # =========================
    def calcRes(self, topLocs, tstLocs, batIds):
        allRecall, allNdcg = 0, 0

        for i in range(len(batIds)):
            preds = list(topLocs[i])
            truth = tstLocs[batIds[i]]

            if truth is None:
                continue

            hits = 0
            dcg = 0

            for item in truth:
                if item in preds:
                    hits += 1
                    dcg += 1 / np.log2(preds.index(item) + 2)

            recall = hits / len(truth)

            max_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(truth), args.topk))])
            ndcg = dcg / max_dcg if max_dcg > 0 else 0

            allRecall += recall
            allNdcg += ndcg

        return allRecall, allNdcg

    # =========================
    # RUN
    # =========================
    def run(self):
        best_recall = 0

        for ep in range(args.epoch):
            train_loss = self.trainEpoch()

            print(f"\nEpoch {ep}/{args.epoch} | Train Loss: {train_loss:.4f}")

            # Always test
            res = self.testEpoch()

            print(f"Recall: {res['Recall']:.6f}, NDCG: {res['NDCG']:.6f}")

            if res['Recall'] > best_recall:
                best_recall = res['Recall']
                print("🔥 New Best Recall!")

        print("\n✅ Training Completed!")


# =========================
# SEED
# =========================
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    seed_it(args.seed)

    log('Start')

    handler = DataHandler()
    handler.LoadData()

    log('Load Data')

    coach = Coach(handler)
    coach.run()