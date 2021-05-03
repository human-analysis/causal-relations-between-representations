# model.py

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics

class Model(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()
        self.save_hyperparameters()
        self.opts = opts

        self.val_dataloader = dataloader.val_dataloader
        self.train_dataloader = dataloader.train_dataloader

        self.model = getattr(models, opts.model_type)(**opts.model_options)
        self.loss_c = getattr(losses, opts.loss_type_c)(**opts.loss_options)
        self.loss_r = getattr(losses, opts.loss_type_r)(**opts.loss_options)
        
        self.acc_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)


    def training_step(self, batch, batch_idx):
        X,Y,labels = batch
        outputs_c,rx,ry,ex,ey,outputs_c_f,_,_ = self.model(X,Y)
        loss1 = self.loss_c(outputs_c, labels)
        loss2 = self.loss_c(outputs_c_f, labels)
        loss3 = self.loss_r(rx, ry)
        loss=self.opts.w1*loss1+self.opts.w2*loss2+self.opts.w3*loss3
        #import pdb;pdb.set_trace()
        acc = self.acc_trn(F.softmax(outputs_c_f, dim=1), labels)        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def validation_step(self, batch, batch_idx):
        X,Y,labels = batch
        outputs_c,rx,ry,ex,ey,outputs_c_f,_,_ = self.model(X,Y)
        loss1 = self.loss_c(outputs_c, labels)
        loss2 = self.loss_c(outputs_c_f, labels)
        loss3 = self.loss_r(rx, ry)
        loss=self.opts.w1*loss1+self.opts.w2*loss2+self.opts.w3*loss3
        acc = self.acc_val(F.softmax(outputs_c_f, dim=1), labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output
    
    def testing_step(self, batch, batch_idx):
        X,Y,labels = batch
        outputs_c,rx,ry,ex,ey,outputs_c_f,_,_ = self.model(X,Y)
        acc = self.acc_tst(F.softmax(outputs_c_f, dim=1), labels)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        if self.opts.scheduler_method is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer, **self.opts.scheduler_options
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]