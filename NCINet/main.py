# main.py

import torch
import random
import config
from model import Model
from train import Trainer
from test import Tester
import utils
import traceback
import sys
from dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():

    # parse the arguments
    args = config.parse_args()
    if (args.ngpu > 0 and torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"

    args.device = torch.device(device)

    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    writer = SummaryWriter('runs/lambda%.2f_idx%d'%(args.w,args.idx))

    dataloader = DataLoader(args)


    # Create Model
    models = Model(args)

    model, criterion, evaluation = models.setup()

    loaders_train = dataloader.create("Train")
    loaders_test = dataloader.create("Test")

    trainer = Trainer (args, model, criterion, evaluation)
    tester = Tester (args, model, criterion, evaluation)


    for epoch in range(int(args.nepochs)):


        print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

        loss_train = trainer.train(epoch, loaders_train, args.reg_proj,writer)

        with torch.no_grad():
            loss_test = tester.test(epoch, loaders_test, args.reg_proj,writer)

        # if epoch % 10 == 0:
        #     args.modelpath = './checkpoint/model_' + str(epoch) + '.pth'
        #     torch.save({'Encoder':model['Encoder'].state_dict(),'Target':model['Target'].state_dict()}, args.modelpath)



if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
