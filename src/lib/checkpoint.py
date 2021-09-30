import os
import torch
from collections import OrderedDict


class CheckPointer:
    """
    class handling model save and load
    save model state dict, optimizer, scheduler, iteration
    """
    def __init__(self, model, logger=None, optimizer=None, scheduler=None, save_dir=None):
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

    def save(self, name, **kwargs):
        save_data = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": self.model.state_dict()
        }
        save_data.update(kwargs)

        save_path = os.path.join(self.save_dir, "{}.pkl".format(name))
        torch.save(save_data, save_path)
        self.logger.info('Saving checkpoint to {}'.format(save_path))

        self.record_last_checkpoint(save_path)

    def load(self, resume_iter=None, best_valid=False, need_resume=False):
        """
        load model from specific checkpoint, or last checkpoint or best valid point for test or not load
        :param resume_iter: resume from specific checkpoint
        :param  best_valid: whether to resume from best valid point
        :return:
        如果存在最后保存的并且需要断点恢复：
            如果指定了哪个epoch 就恢复哪个epoch
            如果要最好的，就恢复最好的
            否则恢复最后那个
        """
        

        if need_resume:
            if resume_iter is None or resume_iter == "":
                if best_valid:
                    save_path = os.path.join(self.save_dir, 'best_valid_dice.pkl')
                else:
                    with open(os.path.join(self.save_dir, 'last_checkpoint'), 'r') as f:
                        save_path = f.read().strip()
            else:
                save_path = os.path.join(self.save_dir, 'model_epoch_{:03d}.pkl'.format(int(resume_iter)))

            self.logger.info('Loading model from {}'.format(save_path))
            checkpoint = torch.load(save_path, map_location="cpu")

            self._load_model_state_dict(checkpoint.pop('model'))
            if self.optimizer: self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

            return checkpoint

        else:
            self.logger.info('No checkpoint, Initializing model')
            return {}

    def _load_model_state_dict(self, loaded_model_state_dict):
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        loaded_model_state_dict = self.strip_prefix_if_present(loaded_model_state_dict, prefix='module.')
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(loaded_model_state_dict)
        else:
            self.model.load_state_dict(loaded_model_state_dict)

    @staticmethod
    def strip_prefix_if_present(state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return state_dict
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict

    def record_last_checkpoint(self, last_checkpoint_path):
        with open(os.path.join(self.save_dir, 'last_checkpoint'), 'w') as f:
            f.write(last_checkpoint_path)

    def has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, 'last_checkpoint'))
