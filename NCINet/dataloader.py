# dataloader.py
import torch
import datasets
import torch.utils.data

class DataLoader:
    def __init__(self, args):
        self.args = args

        self.idx = args.idx
        self.npairs=args.npairs
        self.ndimsx=args.ndimsx
        self.ndimsy=args.ndimsy

        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train

        self.dataset_train = getattr(datasets, self.dataset_train_name)(
            npairs=self.npairs,
            ndimsx=self.ndimsx,
            ndimsy=self.ndimsy,
            idx=self.idx,
            train=True
        )
        self.dataset_test = getattr(datasets, self.dataset_test_name)(
            npairs=self.npairs,
            ndimsx=self.ndimsx,
            ndimsy=self.ndimsy,
            idx=self.idx,
            train=False)

    def create(self, flag=None):
        dataloader = {}
        if flag == "Train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size_train,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        elif flag == "Test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size_test,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True
            )
            return dataloader
