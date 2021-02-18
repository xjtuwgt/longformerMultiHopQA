import pytorch_lightning as pl
from csr_mhqa.data_processing import DataHelper

class HGNDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.config = args

    def setup(self, stage = None):
        self.helper = DataHelper(gz=True, config=self.config)
        self.train_data = self.helper.train_loader
        self.dev_example_dict = self.helper.dev_example_dict
        self.dev_feature_dict = self.helper.dev_feature_dict
        self.dev_data = self.helper.dev_loader

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.dev_data