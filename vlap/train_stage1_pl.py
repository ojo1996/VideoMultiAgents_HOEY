import os
import pytz
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from vlap import QFormerDistiller
from vlap_dataloader import NextQADataset

import mlflow
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

class CombinedModel(nn.Module):
    def __init__(self, hidden_dim=768, t5_model_name='google/flan-t5-xl'):
        super(CombinedModel, self).__init__()
        self.qformer = QFormerDistiller(hidden_dim=hidden_dim)
        self.llm = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.projection_layer = nn.Linear(768, 2048)  # T5の隠れ層サイズに合わせる

        # T5 モデルのパラメータを固定（勾配を計算しない）
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, video_features, question_texts, options_texts):
        _, _, teacher_attn_output = self.qformer(video_features, video_features, question_texts)
        teacher_attn_output = self.projection_layer(teacher_attn_output)
        teacher_attn_output = teacher_attn_output.permute(1, 0, 2)
        
        tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
        question_inputs = tokenizer(question_texts, return_tensors='pt', padding=True, truncation=True).to(teacher_attn_output.device)
        question_features = self.llm.encoder(input_ids=question_inputs.input_ids).last_hidden_state
        
        combined_features = torch.cat((teacher_attn_output, question_features), dim=1)
        
        logits_list = []
        for i in range(combined_features.size(0)):
            options = []
            for j in range(5):
                options.append(options_texts[j][i])
            option_logits = []
            for opt in options:
                choice_ids = tokenizer.encode(opt, return_tensors='pt').to(combined_features.device)
                decoder_input_ids = choice_ids
                outputs = self.llm(encoder_outputs=(combined_features[i].unsqueeze(0), None), decoder_input_ids=decoder_input_ids)
                logits = outputs.logits
                choice_logits = logits[0, -1, :].mean()
                option_logits.append(choice_logits)

            logits_list.append(torch.stack(option_logits))

        return torch.stack(logits_list)

class NextQADataModule(pl.LightningDataModule):
    def __init__(self, csv_file, features_dir, json_file, batch_size):
        super().__init__()
        self.csv_file = csv_file
        self.features_dir = features_dir
        self.json_file = json_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = NextQADataset(self.csv_file, self.features_dir, self.json_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

class CombinedModelLightning(pl.LightningModule):
    def __init__(self, hidden_dim=768, learning_rate=1e-5):
        super(CombinedModelLightning, self).__init__()
        self.model = CombinedModel(hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)

    def forward(self, video_features, question_texts, options_texts):
        return self.model(video_features, question_texts, options_texts)

    def training_step(self, batch, batch_idx):
        video_features = batch['video_features']
        question_texts = batch['question']
        options_texts = batch['options']
        answer = batch['answer']

        logits = self(video_features, question_texts, options_texts)
        loss = self.criterion(logits, answer)
        acc = self.train_acc(logits.argmax(dim=1), answer)  # Accuracy を計算して更新
        self.log('train_loss', loss)
        self.log('train_acc_step', acc, prog_bar=True, on_step=True, on_epoch=False)  # ステップごとのACCログ
        return loss

    def on_train_epoch_end(self):
        # エポックごとのACCを計算してログに出力
        avg_acc = self.train_acc.compute()  # torchmetrics で平均 ACC を計算
        self.log('train_acc_epoch', avg_acc, prog_bar=True, on_epoch=True)
        self.train_acc.reset()  # 次のエポックのためにリセット

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--csv_file', type=str, default='/root/ms1_nas/nextqa/nextqa/train.csv')
    parser.add_argument('--features_dir', type=str, default='/root/ms1_nas/nextqa/NExTVideoFeatures')
    parser.add_argument('--json_file', type=str, default='/root/ms1_nas/nextqa/nextqa/map_vid_vidorID.json')

    args = parser.parse_args()
    
    # Azure ML Workspaceの設定
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.environ.get("TENANT_ID"), force=False)
    workspace = Workspace.from_config('./config.json')
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    

    # MLFlowの設定 https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html
    current_time = datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")
    mlflow_logger = MLFlowLogger(experiment_name='vlap-hpc3', run_name=current_time, tracking_uri=workspace.get_mlflow_tracking_uri(), log_model=True)

    data_module = NextQADataModule(csv_file=args.csv_file, features_dir=args.features_dir, json_file=args.json_file, batch_size=args.batch_size)
    model = CombinedModelLightning(hidden_dim=args.hidden_dim, learning_rate=args.learning_rate)
    
    # if you want resume training
    checkpoint = torch.load('weights/v2/model-epoch=6.ckpt', map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Prepare callbacks
    callbacks=[
        RichProgressBar(),
        ModelCheckpoint(dirpath='./', filename=f'model-{{epoch}}', save_weights_only=True, save_last=False, save_top_k=-1),
        # EarlyStopping(monitor="Rank1ACC", min_delta=0.00, patience=40, verbose=False, mode="max")
    ]

    trainer = pl.Trainer(
        max_epochs   = args.num_epochs,
        accelerator  = 'gpu',
        devices      = [0, 1, 2, 3],
        strategy     = DDPStrategy(find_unused_parameters=True),
        callbacks    = callbacks,
        logger       = mlflow_logger
        )
    trainer.fit(model, data_module)
