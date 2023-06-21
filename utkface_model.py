import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
import torchmetrics as tm
from utkface_utils import plot_metrics
from utkface_data import UTKFaceDataset


class HydraNet(nn.Module):
    def __init__(self, backbone='resnet18', requires_grad=True):
        super().__init__()
        self.backbone = backbone
        
        if self.backbone=='resnet18':
            self.weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.net = torchvision.models.resnet18(weights=self.weights)
            
        if self.backbone=='resnet34':
            self.weights = torchvision.models.ResNet34_Weights.DEFAULT
            self.net = torchvision.models.resnet34(weights=self.weights)
            
        if self.backbone=='resnet50':
            self.weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.net = torchvision.models.resnet50(weights=self.weights)
            
        if self.backbone=='mobilenet_v3_large':
            self.weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
            self.net = torchvision.models.mobilenet_v3_large(weights=self.weights)
            
        if self.backbone=='mobilenet_v3_small':
            self.weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            self.net = torchvision.models.mobilenet_v3_small(weights=self.weights)
            
        if self.backbone=='efficientnet_b0':
            self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            self.net = torchvision.models.efficientnet_b0(weights=self.weights)
            
        if self.backbone=='efficientnet_b1':
            self.weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
            self.net = torchvision.models.efficientnet_b1(weights=self.weights)
            
        if self.backbone=='efficientnet_b2':
            self.weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
            self.net = torchvision.models.efficientnet_b2(weights=self.weights)
            
        if self.backbone=='efficientnet_b3':
            self.weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
            self.net = torchvision.models.efficientnet_b3(weights=self.weights)
        
        for param in self.net.parameters():
            param.requires_grad = requires_grad
        
        if self.backbone[:6]=='resnet':
            clf = self.net.fc
            self.net.fc = nn.Identity()
            self.net.fc1 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(clf.in_features, 
                                     clf.in_features)),
                ('relu', nn.ReLU()),
                ('final_linear', nn.Linear(clf.in_features, 1))
            ]))
            self.net.fc2 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(clf.in_features, 
                                     clf.in_features)),
                ('relu', nn.ReLU()),
                ('final_linear', nn.Linear(clf.in_features, 1))
            ]))
            
        
        if self.backbone[:9]=='mobilenet':
            clf = self.net.classifier
            self.net.classifier = nn.Identity()
            self.net.fc1 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(clf[0].in_features, 
                                     clf[0].out_features)),
                ('hardswish', nn.Hardswish()),
                ('dropout', nn.Dropout(p=0.2, inplace=True)),
                ('final_linear', nn.Linear(clf[0].out_features, 1))
            ]))
            self.net.fc2 = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(clf[0].in_features, 
                                     clf[0].out_features)),
                ('hardswish', nn.Hardswish()),
                ('dropout', nn.Dropout(p=0.2, inplace=True)),
                ('final_linear', nn.Linear(clf[0].out_features, 1))
            ]))


        if self.backbone[:12]=='efficientnet':
            clf = self.net.classifier
            self.net.classifier = nn.Identity()
            self.net.fc1 = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=0.2, inplace=True)),
                ('final_linear', nn.Linear(clf[1].in_features, 1))
            ]))
            self.net.fc2 = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(p=0.2, inplace=True)),
                ('final_linear', nn.Linear(clf[1].in_features, 1))
            ]))
            
    def forward(self, x):
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        
        return age_head, gender_head


class UTKFaceModel(pl.LightningModule):
    def __init__(
        self, 
        train_df: pd.DataFrame=None,
        valid_df: pd.DataFrame=None,
        test_df: pd.DataFrame=None,
        learning_rate: float=1e-3, 
        batch_size: int=32, 
        cm_period: int=5,
        num_workers: int=2,
        backbone='resnet18'
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.mtl_model = HydraNet(backbone)

        self.lr = learning_rate

        self.batch_size = batch_size

        
        self.train_auroc = tm.AUROC(task='binary', compute_on_step=False)
        #self.train_f1 = tm.F1Score(task='binary', compute_on_step=False)
        self.train_acc = tm.Accuracy(task='binary', compute_on_step=False)
        
        self.train_rmse = tm.MeanSquaredError(squared=False, compute_on_step=False)
        self.train_r2 = tm.R2Score(compute_on_step=False)
        
        
        self.val_auroc = tm.AUROC(task='binary', compute_on_step=False)
        #self.val_f1 = tm.F1Score(task='binary', compute_on_step=False)
        self.val_acc = tm.Accuracy(task='binary', compute_on_step=False)
        self.val_cm = tm.ConfusionMatrix(task='binary', compute_on_step=False)
        
        self.val_rmse = tm.MeanSquaredError(squared=False, compute_on_step=False)
        self.val_r2 = tm.R2Score(compute_on_step=False)
        
        
        self.test_auroc = tm.AUROC(task='binary', compute_on_step=False)
        #self.test_f1 = tm.F1Score(task='binary', compute_on_step=False)
        self.test_acc = tm.Accuracy(task='binary', compute_on_step=False)
        self.test_cm = tm.ConfusionMatrix(task='binary', compute_on_step=False)
        
        self.test_rmse = tm.MeanSquaredError(squared=False, compute_on_step=False)
        self.test_r2 = tm.R2Score(compute_on_step=False)

        
        self.cm_period = cm_period
        self.num_workers = num_workers

    def forward(self, x):
        out = self.mtl_model(x.float())
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], 
            lr=self.lr, eps=1e-08)
    
    def train_dataloader(self):
        dataset = UTKFaceDataset(self.train_df, 
                                 transform=self.mtl_model.weights.transforms(
                                     antialias=True))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader

    def val_dataloader(self):
        dataset = UTKFaceDataset(self.valid_df, 
                                 transform=self.mtl_model.weights.transforms(
                                     antialias=True))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader

    def test_dataloader(self):
        dataset = UTKFaceDataset(self.test_df, 
                                 transform=self.mtl_model.weights.transforms(
                                     antialias=True))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
    
    def predict_dataloader(self):
        dataset = UTKFaceDataset(self.test_df, 
                                 transform=self.mtl_model.weights.transforms(
                                     antialias=True))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
    
    def _common_step(self, batch):
        output = self.forward(batch['image'])
        age_output = output[0]
        gender_output = output[1]
        
        gender_loss = nn.BCEWithLogitsLoss()
        age_loss = nn.L1Loss()

        gender_loss = gender_loss(gender_output, batch['gender'].unsqueeze(1).float())
        age_loss = age_loss(age_output, batch['age'].unsqueeze(1).float())
        
        loss = gender_loss + age_loss
        
        return loss, age_output, gender_output
        
    def training_step(self, batch, batch_idx):
        loss, age_logit, gender_logit = self._common_step(batch)
        
        gender_labels = batch['gender'].unsqueeze(1).int()
        age_target = batch['age'].unsqueeze(1)
        
        self.train_auroc.update(gender_logit, gender_labels)
        self.train_acc.update(gender_logit, gender_labels)
        
        self.train_rmse.update(age_logit, age_target)
        self.train_r2.update(age_logit, age_target)
        
        self.log_dict(
            {
                'train/loss': loss, 
                'train/auroc': self.train_auroc, 
                'train/acc': self.train_acc,
                'train/rmse': self.train_rmse,
                'train/r2': self.train_r2
            }, 
            on_epoch=True, 
            on_step=False,
            prog_bar=True
        )
        
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.train_auroc.reset()
        
        self.train_rmse.reset()
        self.train_r2.reset()

        
        self.val_acc.reset()
        self.val_cm.reset()
        self.val_auroc.reset()
        
        self.val_rmse.reset()
        self.val_r2.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, age_logit, gender_logit = self._common_step(batch)
        
        gender_labels = batch['gender'].unsqueeze(1).int()
        age_target = batch['age'].unsqueeze(1)

        self.val_auroc.update(gender_logit, gender_labels)
        self.val_acc.update(gender_logit, gender_labels)
        
        self.val_rmse.update(age_logit, age_target)
        self.val_r2.update(age_logit, age_target)
        
        self.log_dict(
            {
                'val/loss': loss, 
                'val/auroc': self.val_auroc,
                'val/acc': self.val_acc,
                'val/rmse': self.val_rmse,
                'val/r2': self.val_r2
            },
            prog_bar=True
        )

        self.val_cm.update(gender_logit, gender_labels)

    def plot_confusion_matrix(self, df):
        plt.figure(figsize=(4,3))
        ax = sns.heatmap(df, annot=True, cmap='magma', fmt='')
        ax.set_title(f'Confusion Matrix (Epoch {self.current_epoch+1})')
        ax.set_ylabel('True labels')
        ax.set_xlabel('Predicted labels')
        plt.show()

    def on_validation_epoch_end(self):
        if self.current_epoch>0 and (self.current_epoch+1)%self.cm_period==0:
            self.plot_confusion_matrix(
                pd.DataFrame(self.val_cm.compute().detach().cpu().numpy().astype(int)))

    def test_step(self, batch, batch_idx):
        loss, age_logit, gender_logit = self._common_step(batch)
        
        gender_labels = batch['gender'].unsqueeze(1).int()
        age_target = batch['age'].unsqueeze(1)

        self.test_auroc.update(gender_logit, gender_labels)
        self.test_acc.update(gender_logit, gender_labels)
        
        self.test_rmse.update(age_logit, age_target)
        self.test_r2.update(age_logit, age_target)
        
        self.log_dict(
            {
                'test/loss': loss, 
                'test/auroc': self.test_auroc,
                'test/acc': self.test_acc,
                'test/rmse': self.test_rmse,
                'test/r2': self.test_r2
            }
        )

        self.test_cm.update(gender_logit, gender_labels)

    def on_test_epoch_end(self):
        self.plot_confusion_matrix(
            pd.DataFrame(self.test_cm.compute().detach().cpu().numpy().astype(int)))

    def predict_step(self, batch, batch_idx):
        return self(batch['image'])


def train_ptl_model(
    model,
    model_name,
    version,
    epochs,
    print_model=False
):

    if print_model:
        print(model)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir='logs', 
                                             name=model_name,
                                             version=version)

    csv_logger = pl.loggers.CSVLogger(save_dir='logs', 
                                      name=model_name,
                                      version=version)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'logs/{model_name}/{version}/best_ckpt',
        filename=model_name+'_epoch{epoch:02d}-val_loss{val/loss:.2f}',
        auto_insert_metric_name=False,
        monitor='val/loss'
    )

    trainer = pl.Trainer(max_epochs=epochs, logger=[tb_logger, csv_logger], 
                         callbacks=[checkpoint_callback])

    trainer.fit(model)

    plot_metrics(f'logs/{model_name}/{version}/metrics.csv')