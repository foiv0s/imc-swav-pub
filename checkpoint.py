import os
import torch
from model import Model


class Checkpointer:
    def __init__(self, output_dir=None, filename='imc_swav.cpt'):
        self.output_dir = output_dir
        self.epoch = 0
        self.model = None
        self.filename = filename

    def track_new_model(self, model):
        self.model = model

    def restore_model_from_checkpoint(self, cpt_path):
        ckp = torch.load(cpt_path)
        hp = ckp['hyperparams']
        params = ckp['model']
        self.epoch = ckp['epoch']

        self.model = Model(n_classes=hp['n_classes'], encoder_size=hp['encoder_size'], prototypes=hp['prototypes'],
                           project_dim=hp['project_dim'], tau=hp['tau'], eps=hp['eps'])

        model_dict = self.model.state_dict()
        model_dict.update(params)
        params = model_dict
        self.model.load_state_dict(params)

        print("***** CHECKPOINTING *****\n"
              "Model restored from checkpoint.\n"
              "Training epoch {}\n"
              "*************************"
              .format(self.epoch))
        return self.model

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'hyperparams': self.model.hyperparams,
            'epoch': self.epoch
        }

    def update(self, epoch):
        self.epoch = epoch
        cpt_path = os.path.join(self.output_dir, self.filename)
        torch.save(self._get_state(), cpt_path)

    def get_current_position(self):
        return self.epoch
