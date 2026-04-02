import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader


class DataHandler:
    def __init__(self):
        if args.data == 'yelp':
            predir = './Datasets/sparse_yelp/'
        elif args.data == 'lastfm':
            predir = './Datasets/lastFM/'
        elif args.data == 'beer':
            predir = './Datasets/beerAdvocate/'

        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'

    # =========================
    # Load File
    # =========================
    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)

        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)

        return ret

    # =========================
    # Normalize adjacency
    # =========================
    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInv = np.power(degree, -0.5).flatten()
        dInv[np.isinf(dInv)] = 0
        dMat = sp.diags(dInv)
        return mat.dot(dMat).transpose().dot(dMat).tocoo()

    # =========================
    # Create adjacency matrix
    # =========================
    def makeTorchAdj(self, mat):
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))

        mat = sp.vstack([
            sp.hstack([a, mat]),
            sp.hstack([mat.transpose(), b])
        ])

        mat = (mat != 0) * 1.0
        mat = mat + sp.eye(mat.shape[0])
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)

        return t.sparse.FloatTensor(idxs, vals, shape)

    # =========================
    # 🔥 IMPROVED USER SEQUENCES
    # =========================
    def create_user_sequences(self, mat, seq_len=10):
        csr = mat.tocsr()
        user_seq = np.zeros((csr.shape[0], seq_len), dtype=np.int64)

        for u in range(csr.shape[0]):
            items = list(csr[u].indices)

            if len(items) == 0:
                continue

            # 👉 NO SHUFFLE (important for LSTM)
            if len(items) >= seq_len:
                user_seq[u] = items[-seq_len:]   # last interactions
            else:
                # pad with last item
                pad_val = items[-1]
                padded = items + [pad_val] * (seq_len - len(items))
                user_seq[u] = padded

        return user_seq

    # =========================
    # Load Data
    # =========================
    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)

        self.trnMat = trnMat
        args.user, args.item = trnMat.shape

        # Graph
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        # 🔥 Sequence Mapping
        user_seq = self.create_user_sequences(trnMat, seq_len=10)
        self.user_seq_map = t.tensor(user_seq, dtype=t.long)

        print("User sequence shape:", self.user_seq_map.shape)

        # DataLoader
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(
            trnData,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0
        )

        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(
            tstData,
            batch_size=args.tstBat,
            shuffle=False,
            num_workers=0
        )


# =========================
# Train Dataset
# =========================
class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                neg = np.random.randint(args.item)
                if (u, neg) not in self.dokmat:
                    break
            self.negs[i] = neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


# =========================
# Test Dataset
# =========================
class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()

        for i in range(len(coomat.data)):
            u = coomat.row[i]
            v = coomat.col[i]

            if tstLocs[u] is None:
                tstLocs[u] = []
            tstLocs[u].append(v)
            tstUsrs.add(u)

        self.tstUsrs = np.array(list(tstUsrs))
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(
            self.csrmat[self.tstUsrs[idx]].toarray(), [-1]
        )