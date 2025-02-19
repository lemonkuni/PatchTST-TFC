from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# """Two contrastive encoders"""
# class TFC(nn.Module):
#     def __init__(self, configs):
#         super(TFC, self).__init__()

#         encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
#         self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

#         self.projector_t = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#         encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
#         self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

#         self.projector_f = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )


#     def forward(self, x_in_t, x_in_f):
#         """Use Transformer"""
#         x = self.transformer_encoder_t(x_in_t)
#         h_time = x.reshape(x.shape[0], -1)

#         """Cross-space projector"""
#         z_time = self.projector_t(h_time)

#         """Frequency-based contrastive encoder"""
#         f = self.transformer_encoder_f(x_in_f)
#         h_freq = f.reshape(f.shape[0], -1)

#         """Cross-space projector"""
#         z_freq = self.projector_f(h_freq)

#         return h_time, z_time, h_freq, z_freq


from PatchTST import PatchTSTNet
"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        
        # 创建PatchTST配置
        class PatchTSTConfigs:
            def __init__(self):
                # 基础参数
                self.enc_in = 1          # 输入特征维度为1，因为是单通道数据
                self.seq_len = configs.TSlength_aligned  # 使用原始配置的序列长度
                self.pred_len = 24       # 这个可以保持默认值，因为我们只用特征提取部分
                
                # 模型结构参数
                self.e_layers = 3        # encoder层数
                self.n_heads = 8         # 注意力头数
                self.d_model = 128       # 模型维度
                self.d_ff = 256         # 前馈网络维度
                self.dropout = 0.1       # dropout率
                self.fc_dropout = 0.1    # 全连接层dropout率
                self.head_dropout = 0.1  # 输出头dropout率
                
                # Patch相关参数
                self.patch_len = 16      # patch长度
                self.stride = 8          # patch步长
                self.padding_patch = 'end'  # patch填充方式
                
                # 其他参数
                self.individual = False   
                self.revin = True        
                self.affine = True       
                self.subtract_last = False
                self.decomposition = False
                self.kernel_size = 25      # 分解核大小
     

     # 修改预测相关参数为分类参数
        self.num_classes = 6     # 分类类别数
        
        
        # 分类器特定参数
        self.classifier_dropout = 0.1  # 分类器dropout率
        self.use_weighted_loss = False # 是否使用加权损失（处理类别不平衡）
                
        # 创建时间域和频率域的PatchTST编码器
        patchtst_configs = PatchTSTConfigs()
        self.encoder_t = PatchTSTNet(patchtst_configs)
        self.encoder_f = PatchTSTNet(patchtst_configs)
        
        # 保持原有的投影头
        self.projector_t = nn.Sequential(
            nn.Linear(patchtst_configs.d_model, 256),  # 使用d_model作为输入维度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.projector_f = nn.Sequential(
            nn.Linear(patchtst_configs.d_model, 256),  # 使用d_model作为输入维度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        # 调整输入维度 [batch_size, seq_len] -> [batch_size, seq_len, 1]
        x_in_t = x_in_t.unsqueeze(-1)
        x_in_f = x_in_f.unsqueeze(-1)
        
        """Time-based encoder"""
        # 使用PatchTST的backbone部分提取特征
        h_time = self.encoder_t.model(x_in_t.permute(0, 2, 1))  # [batch_size, 1, d_model]
        h_time = h_time.squeeze(1)  # [batch_size, d_model]
        
        """Cross-space projector"""
        z_time = self.projector_t(h_time)  # [batch_size, 128]
        
        """Frequency-based encoder"""
        h_freq = self.encoder_f.model(x_in_f.permute(0, 2, 1))  # [batch_size, 1, d_model]
        h_freq = h_freq.squeeze(1)  # [batch_size, d_model]
        
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)  # [batch_size, 128]
        
        return h_time, z_time, h_freq, z_freq
    

#测试形状的代码
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_files.SleepEEG_Configs import Config

# 创建测试数据和模型
def test_model_shapes():
    # 使用 SleepEEG 的配置
    configs = Config()
    model = TFC(configs)
    classifier = target_classifier(configs)
    
    # 创建模拟输入数据
    batch_size = configs.batch_size  # 128
    x_time = torch.randn(batch_size, configs.TSlength_aligned)  # [128, 178]
    x_freq = torch.randn(batch_size, configs.TSlength_aligned)  # [128, 178]
    
    print("=== 输入数据形状 ===")
    print(f"批次大小: {batch_size}")
    print(f"时间序列长度: {configs.TSlength_aligned}")
    print(f"时间域输入 x_time shape: {x_time.shape}")
    print(f"频率域输入 x_freq shape: {x_freq.shape}")
    
    # 获取模型输出并打印形状
    with torch.no_grad():
        h_time, z_time, h_freq, z_freq = model(x_time, x_freq)
        
        print("\n=== TFC模型各层输出形状 ===")
        print(f"时间域特征 h_time shape: {h_time.shape}")  
        print(f"时间域投影 z_time shape: {z_time.shape}")
        print(f"频率域特征 h_freq shape: {h_freq.shape}")
        print(f"频率域投影 z_freq shape: {z_freq.shape}")
        
        # 测试下游分类器
        combined_features = torch.cat([z_time, z_freq], dim=1)
        predictions = classifier(combined_features)
        
        print("\n=== 下游分类器输出形状 ===")
        print(f"组合特征 combined_features shape: {combined_features.shape}")
        print(f"最终分类输出 predictions shape: {predictions.shape}")

if __name__ == "__main__":
    test_model_shapes()