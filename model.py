import numpy as np
import torch
import torch.nn as nn
from graphs import ResNet, BasicBlock, Bottleneck, Projection, Prototypes, Classifier
from utils import sinkhorn
from costs import entropy, mi


class Model(nn.Module):
    def __init__(self, n_classes, encoder_size=32, prototypes=1000, project_dim=128,
                 tau=0.1, eps=0.05, model_type='resnet18'):
        super(Model, self).__init__()

        self.hyperparams = {
            'n_classes': n_classes,
            'encoder_size': encoder_size,
            'prototypes': prototypes,  # k' number of prototypes
            'project_dim': project_dim,  # projection head's dimension
            'tau': tau,  # Tau parameter of Softmax smoothness (Eq.2)
            'eps': eps,  # epsilon (Eq.3)
            'model_type': model_type
        }

        dummy_batch = torch.zeros((2, 3, encoder_size, encoder_size))

        # encoder that provides multiscale features
        self.encoder = Encoder(encoder_size=encoder_size, model_type=model_type)
        rkhs_1 = self.encoder(dummy_batch)
        self.encoder = nn.DataParallel(self.encoder)
        self.project = Projection(rkhs_1.size(1), project_dim)
        self.prototypes = Prototypes(project_dim, prototypes)
        self.auxhead = Classifier(rkhs_1.size(1), n_classes)  # Classifier
        self.modules_ = [self.encoder.module, self.prototypes, self.project, self.auxhead]
        self._t, self._e = tau, eps
        self._z_bank = []
        self._u_bank = []
        self.counter = 0

    def _encode(self, res_dict, aug_imgs, num_crops):
        res_dict['z'], res_dict['u'] = [], []
        b = aug_imgs[0].size()[0]

        with torch.no_grad():
            # l2 normalization for prototypes
            w = self.prototypes.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.prototypes.weight.copy_(w)

        for aug_imgs_ in aug_imgs:
            emb = self.encoder(aug_imgs_)
            emb_projected = self.project(emb)
            ux = self.prototypes(emb_projected)
            res_dict['z'].append(emb)
            res_dict['u'].append(ux)
        # '''
        swav_loss = []
        for i in range(num_crops[0]):
            with torch.no_grad():
                # Sinkhorn knopp algorithm (Formulation of eq. 3 & 4, based on SWAV implementation)
                # (SWAV URL: https://arxiv.org/abs/2006.09882)
                q = res_dict['u'][i].detach().clone()
                q = torch.exp(q / self._e)
                q = sinkhorn(q.T, 3)  # [-batch_size:]
            for p, px in enumerate(res_dict['u']):
                if p != i:
                    # Equation 1
                    swav_loss.append(entropy(q, torch.softmax(px / self._t, -1)).sum(-1).mean())
        res_dict['swav_loss'] = torch.stack(swav_loss).sum() / len(swav_loss)

        return res_dict

    def encode(self, imgs, res_dict):

        with torch.no_grad():
            z = self.encoder(imgs)
            res_dict['y'] = torch.flatten(self.auxhead(z), 1).argmax(-1)
        return res_dict

    def forward(self, x, nmb_crops=[2], eval_idxs=None, eval_only=False):

        # dict for returning various values
        res_dict = {}
        if eval_only:
            return self.encode(x, res_dict)

        res_dict = self._encode(res_dict, x, nmb_crops)

        '''
        SLT10 contains instances of unlabelled and labelled set..
        Because our method is trained in online mode together with encoder part.
        Due to the ratio of instances between unlabelled (100.000) and labelled set (5000 on training set),
        the training batch size contains a large number of unlabelled instances and a very small number
        of training instances. 
        Hence, we use a short temporary bank to store representations (z) till we can reach the same number 
        as the batch size of labelled instances.
        Afterwards, the membanks are used to train the classifier. 
        '''

        # The classifier is trained only on labelled training set
        # This actually applies only on STL10 where there is an unlabelled and labelled set
        # We train and evaluate the classifier only on labelled set, hence eval_idxs is a boolean array
        # for performing training only on labelled idxs
        if eval_idxs is None:
            eval_idxs = torch.ones(res_dict['z'][0].size(0), dtype=torch.bool)

        # Below lists are membanks to store embeddings and u probability distributions.
        # These are only used for training on STL10.
        # As it is mentioned above classifier is trained only on labelled set of STL10
        # We just collect the labelled embedding representations till the collection reaches equal number to
        # the training batch size
        # This happens because a batch size on STL10 contains instances of label and unlabelled set
        # Hence each batch size contains a very small number of training instances and we use this collection
        # In order to maintain the batch size of classifier on average equals to the encoder batch size
        # '''

        with torch.no_grad():
            self._z_bank.append(torch.cat([z[eval_idxs].unsqueeze(1).detach() for z in res_dict['z']], 1))
            self._u_bank.append(torch.cat([ux[eval_idxs].unsqueeze(1).detach() for ux in res_dict['u']], 1))
        b = res_dict['z'][0].size(0)
        self.counter += eval_idxs.sum().item()
        mi_loss, lgt_reg, res_dict['y'] = [], [], None
        if self.counter >= b:
            Z = torch.cat(self._z_bank, 0).unbind(1)
            Y = [self.auxhead(z) for z in Z]
            U = torch.unbind(torch.cat(self._u_bank), 1)
            for j, py_j in enumerate(Y):
                for u, pu_i in enumerate(U):
                    mi_loss_, lgt_reg_ = mi(py_j, pu_i / self._t)  # Equation 6
                    mi_loss.append(mi_loss_), lgt_reg.append(lgt_reg_)
            res_dict['y'] = torch.flatten(Y[0], 1).argmax(-1)
            self.reset_membank_list()
        zero = torch.tensor([0], device=x[0].device.type)
        res_dict['mi_loss'] = torch.stack(mi_loss).mean() if len(mi_loss) > 0 else zero
        res_dict['lgt_reg'] = torch.stack(lgt_reg).mean() if len(lgt_reg) > 0 else zero

        return res_dict

    # Reset the membanks, this actually applies only on STL10 training because of the unlabelled set
    def reset_membank_list(self):
        self._z_bank, self._u_bank = [], []
        self.counter = 0


class Encoder(nn.Module):
    def __init__(self, encoder_size=32, model_type='resnet18'):
        super(Encoder, self).__init__()

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        inplanes = 64
        if encoder_size == 32:
            conv1 = nn.Sequential(nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(inplanes),
                                  nn.ReLU(inplace=True))
        elif encoder_size == 96 or encoder_size == 64:
            conv1 = nn.Sequential(nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(inplanes),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))

        if model_type == 'resnet18':
            # ResNet18 block
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], conv1)
        elif model_type == 'resnet34':
            self.model = ResNet(BasicBlock, [3, 4, 6, 3], conv1)
        elif model_type == 'resnet50':
            self.model = ResNet(Bottleneck, [3, 4, 6, 3], conv1)
        else:
            raise RuntimeError("Wrong model type")

        print(self.get_param_n())

    def get_param_n(self):
        w = 0
        for p in self.model.parameters():
            w += np.product(p.shape)
        return w

    def forward(self, x):
        return torch.flatten(self.model(x), 1)
