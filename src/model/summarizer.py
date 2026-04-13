import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class MultiHeadQueryAttention(nn.Module):
    def __init__(self, hidden_size, query_size, num_heads=4):
        super(MultiHeadQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query_proj = nn.Linear(query_size, hidden_size)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, q):
        """
        h: LSTM outputs, shape (batch_size, seq_len, hidden_size)
        q: Query embedding, shape (batch_size, query_size)
        """
        batch_size, seq_len, hidden_size = h.size()
        
        # Project and transform query into 'num_heads'
        q_proj = self.query_proj(q).unsqueeze(1) # (batch, 1, hidden_size)
        
        # Standard Multi-head Attention logic where 'q' is the query 
        # and 'h' are both keys and values
        query = self.q_linear(q_proj).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_linear(h).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_linear(h).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        # Energy: tensor @ tensor.transpose
        energy = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(energy, dim=-1) # (batch, heads, 1, seq_len)
        
        # Context
        out = torch.matmul(attn, values) # (batch, heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, hidden_size)
        
        # We want attention weights per frame for the final scoring
        # Average attention across heads
        attn_weights = attn.mean(dim=1).transpose(1, 2) # (batch, seq_len, 1)
        
        return attn_weights, self.out_proj(out)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(scores)
        out = torch.bmm(attn, v)
        return self.norm(out + x) # Residual + Norm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.encoding[:, :x.size(1), :]

class VideoSummarizer(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=3, query_size=384, transformer_heads=8):
        super(VideoSummarizer, self).__init__()
        
        # 0. Initial Projection for CLIP Features
        self.feat_proj = nn.Linear(input_size, hidden_size*2)
        
        # 1. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size*2)
        
        # 2. Transformer Encoder Layer (Global Context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size*2, 
            nhead=transformer_heads, 
            dim_feedforward=hidden_size*4,
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. 3-Layer Residual Bi-LSTM (Temporal Dynamics)
        self.lstms = nn.ModuleList([
            nn.LSTM(hidden_size*2, 
                    hidden_size, 
                    num_layers=1, 
                    batch_first=True, 
                    bidirectional=True)
            for i in range(num_layers)
        ])
        
        self.lstm_norms = nn.ModuleList([nn.LayerNorm(hidden_size*2) for _ in range(num_layers)])
                            
        lstm_out_size = hidden_size * 2 
        
        # 4. Multi-Head Query Attention (Semantic Alignment)
        self.query_attention = MultiHeadQueryAttention(hidden_size=lstm_out_size, query_size=query_size)
        
        # 5. Score Regressor
        self.score_regressor = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, q):
        """
        x: CLIP features, shape (batch, seq_len, input_size)
        q: Query embedding, shape (batch, query_size)
        """
        # 0. Project and Encode Position
        h = self.feat_proj(x)
        h = self.pos_encoder(h)
        
        # 1. Transformer Global Context
        h = self.transformer_encoder(h)
        
        # 2. Residual Stacked LSTM
        for i, (lstm, norm) in enumerate(zip(self.lstms, self.lstm_norms)):
            h_next, _ = lstm(h)
            h = norm(h_next + h) # Residual connection with LayerNorm
        
        # 3. Multi-Head Query Attention
        attn_weights, context = self.query_attention(h, q)
        
        # 4. Feature weighting
        attended_features = h * attn_weights
        
        # 5. Score Regressor
        scores = self.score_regressor(attended_features)
        
        return scores.squeeze(-1)

class QueryEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        except Exception:
            print("Cache miss — downloading MiniLM from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, query_text):
        """
        Converts a text query into an embedding vector.
        """
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling for sentence embedding
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.cpu().numpy()
