import torch
import numpy as np
from omegaconf import OmegaConf
from rnn_model import GRUDecoder
from dataset import BrainToTextDataset, train_test_split_indicies
from torch.utils.data import DataLoader
from data_augmentations import gauss_smooth

class PhonemeDecoder:
    """
    用於推論並輸出音素序列的類別
    """
    
    def __init__(self, checkpoint_path, args_path, device='cuda: 0'):
        """
        checkpoint_path: 模型 checkpoint 路徑
        args_path: args. yaml 路徑
        device: 運算裝置
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = OmegaConf.load(args_path)
        
        # 初始化模型
        self.model = GRUDecoder(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            n_classes=self.args['dataset']['n_classes'],
            rnn_dropout=0.0,  # 推論時關閉 dropout
            input_dropout=0.0,
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
        )
        
        # 載入權重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 載入音素對照表（39 個音素 + blank）
        self.phoneme_list = self._get_phoneme_list()
        
    def _get_phoneme_list(self):
        """
        返回音素列表（index 0 = blank for CTC）
        """
        # 根據 CMU 音素集，這是標準的 39 音素 + blank
        phonemes = [
            'BLANK',  # CTC blank token (index 0)
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL'
        ]
        return phonemes
    
    def decode_logits_to_phonemes(self, logits, seq_len):
        """
        將 logits 解碼為音素序列
        
        Args:
            logits:  模型輸出 (time_steps, n_classes)
            seq_len: 有效序列長度
            
        Returns:
            phoneme_indices: 音素索引序列 (去除重複和 blank)
            phoneme_sequence: 音素字串序列
        """
        # Greedy decoding:  取每個時間步機率最高的音素
        predictions = torch.argmax(logits[: seq_len], dim=-1)
        
        # CTC 解碼：去除連續重複
        predictions = torch.unique_consecutive(predictions)
        
        # 移除 blank token (index 0)
        predictions = predictions. cpu().numpy()
        phoneme_indices = [idx for idx in predictions if idx != 0]
        
        # 轉換為音素字串
        phoneme_sequence = [self.phoneme_list[idx] for idx in phoneme_indices]
        
        return phoneme_indices, phoneme_sequence
    
    def predict(self, features, day_idx, n_time_steps):
        """
        對神經訊號進行推論，輸出音素序列
        
        Args:
            features:  神經特徵 (batch_size, time_steps, 512)
            day_idx: 日期索引
            n_time_steps: 每個樣本的有效時間步數
            
        Returns:
            results: 包含音素序列的列表
        """
        self.model.eval()
        results = []
        
        with torch.no_grad():
            features = features.to(self.device)
            day_idx = day_idx.to(self.device)
            
            # 計算調整後的序列長度（考慮 patch_size 和 stride）
            patch_size = self.args['model']['patch_size']
            patch_stride = self.args['model']['patch_stride']
            adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
            
            # 模型推論
            logits = self.model(features, day_idx)
            
            # 對每個樣本解碼
            for i in range(logits.shape[0]):
                seq_len = adjusted_lens[i]. item()
                phoneme_indices, phoneme_sequence = self. decode_logits_to_phonemes(
                    logits[i], seq_len
                )
                
                results.append({
                    'phoneme_indices': phoneme_indices,
                    'phoneme_sequence': phoneme_sequence,
                    'phoneme_string': ' '.join(phoneme_sequence)
                })
        
        return results


# ============================================
# 使用範例
# ============================================
if __name__ == "__main__": 
    import h5py
    
    # 路徑設定（請根據你的環境修改）
    checkpoint_path = '../data/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint'
    args_path = '../data/t15_pretrained_rnn_baseline/checkpoint/args.yaml'
    data_path = '../data/hdf5_data_final/t15.2023. 08.13/data_val.hdf5'
    
    # 初始化解碼器
    decoder = PhonemeDecoder(checkpoint_path, args_path)
    
    # 載入測試資料
    with h5py.File(data_path, 'r') as f:
        # 取得一筆資料
        trial_keys = list(f. keys())
        trial = f[trial_keys[0]]
        
        features = torch.tensor(trial['inputFeatures'][:]).unsqueeze(0)  # (1, T, 512)
        n_time_steps = torch.tensor([features.shape[1]])
        day_idx = torch.tensor([0])  # 假設是第一天的資料
    
    # 推論
    results = decoder.predict(features, day_idx, n_time_steps)
    
    print("=" * 50)
    print("音素序列輸出:")
    print(f"音素索引: {results[0]['phoneme_indices']}")
    print(f"音素序列: {results[0]['phoneme_sequence']}")
    print(f"音素字串:  {results[0]['phoneme_string']}")