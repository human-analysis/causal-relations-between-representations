# model.py
import models
import losses
import evaluate
from torch import nn

class Model:
    def __init__(self, args):
        self.ngpu = args.ngpu
        self.device = args.device


        self.model_type_E = args.model_type_e
        self.model_type_T = args.model_type_t

        self.model_options_E = {"indim":args.indim,"outdim":args.outdim,"reg":args.reg,"hdlayers":args.nunits}
        self.model_options_T = args.model_options_T

        self.loss_type_E = args.loss_type_e
        self.loss_type_T = args.loss_type_t
        self.loss_type_R = args.loss_type_r

        self.loss_options_E = args.loss_options_E
        self.loss_options_T = args.loss_options_T
        self.loss_options_R = args.loss_options_R

        self.evaluation_type_T = args.evaluation_type_t
        self.evaluation_options_T = args.evaluation_options_T

    def setup(self):

        model_E = getattr(models, self.model_type_E)(**self.model_options_E)
        model_T = getattr(models, self.model_type_T)(**self.model_options_T)

        criterion_E = getattr(losses, self.loss_type_E)(**self.loss_options_E)
        criterion_T = getattr(losses, self.loss_type_T)(**self.loss_options_T)
        criterion_r = getattr(losses, self.loss_type_R)(**self.loss_options_R)

        evaluation_T = getattr(evaluate, self.evaluation_type_T)(
            **self.evaluation_options_T)

        if self.ngpu > 1:
            model_E = nn.DataParallel(model_E, device_ids=list(range(self.ngpu)))
            model_T = nn.DataParallel(model_T, device_ids=list(range(self.ngpu)))

        model_E = model_E.to(self.device)
        model_T = model_T.to(self.device)
        criterion_T = criterion_T.to(self.device)

        model ={}
        model['Encoder'] = model_E
        model['Target'] = model_T

        criterion = {}
        criterion['Encoder'] = criterion_E
        criterion['Target'] = criterion_T
        criterion['regression'] = criterion_r

        evaluation = {}
        evaluation['Target'] = evaluation_T

        return model, criterion, evaluation
