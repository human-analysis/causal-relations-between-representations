# test.py
import torch
import plugins


class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.total_classes = args.total_classes
        self.nclasses_A = args.nclasses_a
        self.nclasses_T = args.nclasses_t

        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.device = args.device
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size_test

        self.w=args.w

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


        self.params_loss = ['Loss_Target','Loss_r', 'Accuracy_Target']#

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss_Target': {'dtype': 'running_mean'},
            'Loss_r': {'dtype': 'running_mean'},
            'Accuracy_Target': {'dtype': 'running_mean'},
           }

        self.monitor.register(self.params_monitor)

        self.print_formatter = 'Test [%d/%d][%d/%d] '
        for item in self.params_loss:
            self.print_formatter += item + " %.4f "

        self.losses = {}

    def model_eval(self):
        self.model['Encoder'].eval()
        self.model['Target'].eval()

    def test(self, epoch, dataloader, reg,writer):

        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()


        for i, (X, Y, labels, sensitives) in enumerate(dataloader):

            ############################
            # Evaluate Network
            ############################
            batch_size=X.size(0)
            self.labels_org.resize_(labels.size()).copy_(labels)
            self.sensitives_org.resize_(sensitives.size()).copy_(sensitives)
            labels = torch.zeros(batch_size, self.total_classes).scatter_(1, labels.unsqueeze(1).long(), 1)
            sensitives = torch.zeros(batch_size, self.total_classes).scatter_(1, sensitives.unsqueeze(1).long() +
                                                                     self.total_classes-self.nclasses_A, 1)
            self.X = X.to(self.device)
            self.Y = Y.to(self.device)
            self.labels = labels.to(torch.float).to(self.device)

            self.sensitives = sensitives.to(torch.float).to(self.device)
            in_c,self.outputs_E ,rx,ry = self.model['Encoder'](self.X,self.Y)

            self.outputs_E= self.outputs_E /torch.norm(self.outputs_E,dim=1).unsqueeze(1).repeat(1,3)

            #### adversarial loss ####
            # loss_a = self.criterion['Encoder'](self.outputs_E, self.sensitives, self.labels, reg,
            #                                                      self.device,
            #                                                      self.args.sigma)

            #### regression loss ####
            loss_r=self.criterion['regression'](rx/batch_size,ry/batch_size)

            #### classification loss ####
            outputs_c = self.model['Target'](self.outputs_E,in_c)
            loss_c = self.criterion['Target'](outputs_c.squeeze(), self.labels_org.long())



            acc = self.evaluation['Target'](outputs_c.squeeze(), self.labels_org.long())
            acc= acc.item()

            #loss_a = loss_a.item()

            loss_r=loss_r.item()
            loss_c = loss_c.item()

            self.losses['Loss_Target'] = loss_c
            self.losses['Loss_r'] = loss_r
            self.losses['Accuracy_Target'] = acc
            self.monitor.update(self.losses, batch_size)


            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                [self.losses[key] for key in self.params_monitor]))


        loss = self.monitor.getvalues()

        #### visualization ####

        if epoch % 10 == 0:

            writer.add_scalar('Loss_Target test',
                              loss['Loss_Target'],
                              epoch),
            writer.add_scalar('Loss_r test',
                              loss['Loss_r'],
                              epoch),
            writer.add_scalar('testings acc',
                              loss['Accuracy_Target'],
                              epoch)

        return loss

