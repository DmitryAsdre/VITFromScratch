import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class PatchMebeddings(nn.Module):
    def __init__(self, 
                 image_size : int,
                 patch_size : int,
                 num_channels : int,
                 hidden_size : int):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, 
                x : torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Embeddings(nn.Module):
    def __init__(self,
                 image_size : int,
                 patch_size : int,
                 num_channels : int,
                 hidden_size : int,
                 hidden_dropout_prob : float):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        self.patch_embeddings = PatchMebeddings(image_size,
                                                patch_size,
                                                num_channels,
                                                hidden_size)
        #cls embededing
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        #position embedding
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    
class AttentionHead(nn.Module):
    def __init__(self, 
                 hidden_size : int,
                 attention_head_size : int,
                 dropout : float,
                 bias : bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.bias = bias
        
        self.query = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.key = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.value = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, 
                x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #(batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_hidden_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention_scores = query @ (key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_scores)
        
        attention_output = attention_probs @ value
        
        return attention_output,\
               attention_probs

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_attention_heads : int,
                 hidden_size : int,
                 hidden_dropout_prob : float, 
                 qkv_dropout : float,
                 qkv_bias : bool):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.qkv_dropout = qkv_dropout
        self.qkv_bias = qkv_bias
        
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                self.qkv_dropout,
                self.qkv_bias
            )
            self.heads.append(head)
            
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(self.hidden_dropout_prob)
        
    def forward(self,
                x : torch.Tensor,
                output_attentions : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        if not output_attentions:
            return attention_output, None
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=-1)
            return attention_output, attention_probs
        
class NewGELUActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                input : torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    
class MLP(nn.Module):
    def __init__(self,
                 hidden_size : int,
                 intermediate_size : int,
                 hidden_dropout_prob : float):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        
        self.dense1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.activation = NewGELUActivation()
        self.dense2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, 
                 num_attention_heads : int,
                 hidden_size : int,
                 hidden_dropout_prob : float,
                 qkv_dropout : float,
                 qkv_bias : bool,
                 intermediate_size : int):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.qkv_dropout = qkv_dropout
        self.qkv_bias = qkv_bias
        self.intermediate_size = intermediate_size
        
        
        self.attention = MultiHeadAttention(self.num_attention_heads,
                                            self.hidden_size,
                                            self.hidden_dropout_prob,
                                            self.qkv_dropout,
                                            self.qkv_bias)
        self.layernorm1 = nn.LayerNorm(self.hidden_size)
        self.mlp = MLP(self.hidden_size,
                       self.intermediate_size, 
                       self.hidden_dropout_prob)
        self.layernorm2 = nn.LayerNorm(self.hidden_size)
        
    def forward(self,
                x : torch.Tensor,
                output_attentions : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, attention_probs = \
            self.attention(self.layernorm1(x), output_attentions=output_attentions)
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm2(x))
        
        x = x + mlp_output
        
        if not output_attentions:
            return x, None
        else:
            return x, attention_probs
        
class VITEncoder(nn.Module):
    def __init__(self,
                 num_hidden_layers : int,
                 num_attention_heads : int,
                 hidden_size : int,
                 hidden_dropout_prob : float,
                 qkv_dropout : float,
                 qkv_bias : bool,
                 intermediate_size : int):
        super().__init__()
        
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.qkv_dropout = qkv_dropout
        self.qkv_bias = qkv_bias
        self.intermediate_size = intermediate_size
        
        self.blocks = nn.ModuleList([])
        for _ in range(self.num_hidden_layers):
            block = EncoderBlock(self.num_attention_heads,
                                 self.hidden_size,
                                 self.hidden_dropout_prob,
                                 self.qkv_dropout,
                                 self.qkv_bias,
                                 self.intermediate_size)
            self.blocks.append(block)
            
    def forward(self,
                x : torch.Tensor,
                output_attentions : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
                
                
        if not output_attentions:
            return x, None
        else:
            return x, all_attentions
        
class VITForClassification(nn.Module):
    def __init__(self,
                 image_size : int,
                 num_classes : int,
                 patch_size : int,
                 num_channels : int,
                 num_hidden_layers : int,
                 num_attention_heads : int,
                 hidden_size : int,
                 hidden_dropout_prob : float,
                 qkv_dropout : float,
                 qkv_bias : bool,
                 intermediate_size : int):
        super().__init__()
        
        self.image_size = image_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.qkv_dropout = qkv_dropout
        self.qkv_bias = qkv_bias 
        self.intermediate_size = intermediate_size
        
        self.embedding = Embeddings(self.image_size,
                                    self.patch_size, 
                                    self.num_channels, 
                                    self.hidden_size,
                                    self.hidden_dropout_prob)
        self.encoder = VITEncoder(self.num_hidden_layers,
                                  self.num_attention_heads,
                                  self.hidden_size,
                                  self.hidden_dropout_prob,
                                  self.qkv_dropout,
                                  self.qkv_bias,
                                  self.intermediate_size)
        
        self.classifier = nn.Linear(self.hidden_size,
                                    self.num_classes)
        #self.apply(self._init_weights)
        
    def forward(self,
                x : torch.Tensor,
                output_attentions : bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # CLS token for classification
        logits = self.classifier(encoder_output[:, 0])
        
        if not output_attentions:
            return logits, None
        else:
            return logits, all_attentions