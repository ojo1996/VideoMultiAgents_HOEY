import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

# QFormer-Distillerクラスの定義
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

class QFormerDistiller(nn.Module):
    def __init__(self, hidden_dim=768):
        super(QFormerDistiller, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # BERTモデルの設定
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_config.is_decoder = True  # デコーダーとして設定
        
        # 教師QFormerと生徒QFormerの初期化
        self.qformer_teacher = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)  # 教師モデル
        self.qformer_student = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)  # 生徒モデル
        
        # 質問の学習可能なクエリ
        self.learnable_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))  # 学習可能なクエリの初期化
        
        # ビデオ特徴量の次元を合わせるための線形層
        self.feature_projector = nn.Linear(1536, self.hidden_dim)

        # クロスアテンション層の定義
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8)
        
        # デコーダー（生徒モデルの出力専用）
        self.decoder = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 損失関数
        self.distillation_loss = nn.MSELoss()

        # トークナイザーの初期化
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, all_video_features, selected_video_features, question_text):
        """
        Args:
            all_video_features (Tensor): すべてのビデオフレームの特徴 (batch_size, num_frames, hidden_dim)
            selected_video_features (Tensor): 選択されたフレームの特徴 (batch_size, num_selected_frames, hidden_dim)
            question_text (List[str]): 質問のテキストデータ

        Returns:
            decoded_student_output (Tensor): 生徒モデルのデコーダー出力
            distill_loss (Tensor): 蒸留損失
            teacher_attn_output (Tensor): 教師モデルのクロスアテンション出力
        """
        # ビデオ特徴量をプロジェクションして次元を調整
        projected_all_video_features = self.feature_projector(all_video_features)  # (batch_size, num_frames, 768)
        projected_selected_video_features = self.feature_projector(selected_video_features)  # (batch_size, num_selected_frames, 768)

        # 質問テキストのトークナイズとエンコード
        inputs = self.tokenizer(question_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # バッチサイズを動的に取得して学習可能なクエリを拡張
        batch_size = all_video_features.size(0)
        learnable_query = self.learnable_query.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)

        # 教師QFormerによる処理（すべてのフレーム特徴を使用）
        teacher_inputs = torch.cat((learnable_query, projected_all_video_features), dim=1)  # (batch_size, 1 + num_frames, hidden_dim)
        teacher_outputs = self.qformer_teacher(
            inputs_embeds=teacher_inputs,
            attention_mask=None,
            return_dict=True
        )

        # 生徒QFormerによる処理（選択されたフレーム特徴を使用）
        student_inputs = torch.cat((learnable_query, projected_selected_video_features), dim=1)  # (batch_size, 1 + num_selected_frames, hidden_dim)
        student_outputs = self.qformer_student(
            inputs_embeds=student_inputs,
            attention_mask=None,
            return_dict=True
        )

        # 教師モデルのクロスアテンションの適用
        teacher_attn_output, _ = self.cross_attention(
            query=teacher_outputs.last_hidden_state.transpose(0, 1),
            key=projected_all_video_features.transpose(0, 1),
            value=projected_all_video_features.transpose(0, 1)
        )

        # 生徒モデルのクロスアテンションの適用
        student_attn_output, _ = self.cross_attention(
            query=student_outputs.last_hidden_state.transpose(0, 1),
            key=projected_selected_video_features.transpose(0, 1),
            value=projected_selected_video_features.transpose(0, 1)
        )

        # 生徒モデルの出力をデコーダーを通して変換
        decoded_student_output = self.decoder(student_attn_output.transpose(0, 1))  # (batch_size, num_selected_frames, hidden_dim)

        # 蒸留損失の計算（デコーダー出力と教師モデルの出力を比較）
        distill_loss = self.distillation_loss(decoded_student_output, teacher_attn_output.transpose(0, 1))

        return decoded_student_output, distill_loss, teacher_attn_output


class FramePrompter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_segments=4):
        super(FramePrompter, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.num_segments = num_segments

    def forward(self, video_features):
        """
        Args:
            video_features (Tensor): ビデオフレームの特徴 (batch_size, num_frames, feature_dim)
        
        Returns:
            selected_features (Tensor): 選択されたフレームの特徴 (batch_size, selected_frames, feature_dim)
        """
        # Mean-pooling to aggregate frame features
        mean_pooled_features = video_features.mean(dim=1)  # (batch_size, feature_dim)

        # Encode the mean-pooled features
        encoded_features = F.relu(self.encoder(mean_pooled_features))  # (batch_size, hidden_dim)

        # Apply Gumbel-Softmax to select segments
        gumbel_logits = torch.log(encoded_features + 1e-10)  # (batch_size, hidden_dim)
        gumbel_mask = F.gumbel_softmax(gumbel_logits, tau=1, hard=True)  # (batch_size, hidden_dim)

        # Use Gumbel mask to select frames
        selected_features = video_features * gumbel_mask.unsqueeze(1)  # (batch_size, num_frames, feature_dim)
        
        return selected_features