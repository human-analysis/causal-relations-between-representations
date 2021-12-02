# train.py

import time
import torch
import torch.optim as optim
import plugins



class Trainer:
    def __init__(self, args, model, criterion, evaluation):

        self.args = args
        # self.r = args.r
        self.total_classes = args.total_classes
        self.nclasses_A = args.nclasses_a
        self.nclasses_T = args.nclasses_t

        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation


        # self.save_results = args.save_results

        # self.env = args.env
        # self.port = args.port
        # self.dir_save = args.save_dir
        # self.log_type = args.log_type

        self.device = args.device
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size_train

        # self.resolution_high = args.resolution_high
        # self.resolution_wide = args.resolution_wide

        self.lr_e = args.learning_rate_e
        self.optim_method_e = args.optim_method_e
        self.optim_options_e = args.optim_options_e
        self.scheduler_method_e = args.scheduler_method_e
        self.scheduler_options_e = args.scheduler_options_e

        self.w=args.w


        self.optimizer = {}

        self.optimizer['Encoder'] = getattr(optim, self.optim_method_e)(
            filter(lambda p: p.requires_grad, list(self.model['Encoder'].parameters())+list(self.model['Target'].parameters())),
            lr=self.lr_e, **self.optim_options_e)


        self.scheduler = {}
        if self.scheduler_method_e is not None:
            self.scheduler['Encoder'] = getattr(optim.lr_scheduler, self.scheduler_method_e)(
                self.optimizer['Encoder'], **self.scheduler_options_e
            )


        # for classification
        self.X = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.Y = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.sensitives_org = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.labels_org = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )

        self.sensitives = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )
        self.labels = torch.zeros(
            self.batch_size,
            dtype=torch.float,
            device=self.device
        )


        self.params_loss = ['Loss', 'P_M*A','Loss_Target','Loss_r','Accuracy_Target']

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss': {'dtype': 'running_mean'},
            'P_M*A': {'dtype': 'running_mean'},
            'Loss_Target': {'dtype': 'running_mean'},
            'Loss_r': {'dtype': 'running_mean'},
            'Accuracy_Target': {'dtype': 'running_mean'},
           }

        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in self.params_loss:
            self.print_formatter += item + " %.4f "


        self.evalmodules = []
        self.losses = {}


    def model_train(self):

        self.model['Encoder'].train()
        self.model['Target'].train()



    def train(self, epoch, dataloader, reg,writer):
        dataloader = dataloader['train']
        self.monitor.reset()
        self.model_train()

        for i, (X, Y, labels, sensitives) in enumerate(dataloader):

            ############################
            # Update network
            ############################
            batch_size=X.size(0)


            self.labels_org.resize_(labels.size()).copy_(labels)
            self.sensitives_org.resize_(sensitives.size()).copy_(sensitives)

            #### ARL-used label####
            labels = torch.zeros(batch_size, self.total_classes).scatter_(1, labels.unsqueeze(1).long(), 1)
            sensitives = torch.zeros(batch_size, self.total_classes).scatter_(1, sensitives.unsqueeze(1).long() +
                                                                     self.total_classes-self.nclasses_A, 1)
            self.labels = labels.to(torch.float).to(self.device)
            self.sensitives = sensitives.to(torch.float).to(self.device)

            self.X = X.to(self.device)
            self.Y = Y.to(self.device)

            in_c,self.outputs_E,rx,ry= self.model['Encoder'](self.X,self.Y)

            self.outputs_E= self.outputs_E /torch.norm(self.outputs_E,dim=1).unsqueeze(1).repeat(1,self.outputs_E.shape[1])

            #### adversarial loss ####
            loss_a = self.criterion['Encoder'](self.outputs_E, self.sensitives, self.labels, reg,
                                                                 self.device,
                                                                 self.args.sigma)

            #### regression loss ####
            loss_r=self.criterion['regression'](rx/batch_size,ry/batch_size)

            #### classification loss ####
            outputs_c = self.model['Target'](self.outputs_E,in_c)
            loss_c = self.criterion['Target'](outputs_c, self.labels_org.long())

            loss=loss_a*self.w+loss_r+loss_c

            self.optimizer['Encoder'].zero_grad()
            loss.backward()
            self.optimizer['Encoder'].step()

            acc = self.evaluation['Target'](outputs_c, self.labels_org.long())
            acc = acc.item()

            loss_a = loss_a.item()
            loss_r=loss_r.item()
            loss_c=loss_c.item()
            loss = loss.item()

            ############################
            #   Evaluating
            ############################

            self.losses['Loss'] = loss
            self.losses['P_M*A'] = loss_a
            self.losses['Loss_Target'] = loss_c
            self.losses['Loss_r'] = loss_r
            self.losses['Accuracy_Target'] = acc

            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
            [epoch + 1, self.nepochs, i, len(dataloader)] +
            [self.losses[key] for key in self.params_monitor]))


        loss = self.monitor.getvalues()

        #### visualization ####

        if epoch % 10 == 0:
            writer.add_scalar('training loss',
                              loss['Loss'],
                              epoch),
            writer.add_scalar('P_M*A train',
                              loss['P_M*A'],
                              epoch),
            writer.add_scalar('Loss_Target train',
                              loss['Loss_Target'],
                              epoch),
            writer.add_scalar('Loss_r train',
                              loss['Loss_r'],
                              epoch),
            writer.add_scalar('training acc',
                              loss['Accuracy_Target'],
                              epoch)

        if self.scheduler_method_e is not None:
            self.scheduler['Encoder'].step()

        return loss
