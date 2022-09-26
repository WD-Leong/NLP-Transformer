import torch
import torch.nn.functional as F

# Multi-Head Attention Layer. #
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, neg_infty=-1.0e9):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.neg_infty = neg_infty

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wc = torch.nn.Linear(
            d_model, d_model, bias=False)
    
    def dot_attention(self, q, k, mask=None):
        lq = list(torch._shape_as_tensor(q))[2]
        lk = list(torch._shape_as_tensor(k))[2]
        
        # Head dimension. #
        dk = list(torch._shape_as_tensor(k))[-1]
        dk = torch.as_tensor(dk, dtype=torch.float32)
        
        # Multiplicative Attention. #
        matmul_qk = torch.matmul(
            q, torch.transpose(k, 2, 3))
        
        # Scale multiplicative attention mechanism. #
        attn_logits = matmul_qk * torch.rsqrt(dk)
        
        # Add the mask to the attention mechanism. #
        if mask is not None:
            attn_mask = (mask * self.neg_infty)
        else:
            attn_mask = torch.zeros(1, 1, lq, lk)
        
        if torch.cuda.is_available():
            attn_mask = attn_mask.to("cuda")
        
        attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=3)
        return attn_weights

    def split_heads(self, x):
        input_dims = list(torch._shape_as_tensor(x))
        batch_size = input_dims[0]
        input_len  = input_dims[1]
        depth_size = int(input_dims[2] / self.n_heads)
        
        output_shape = tuple(
            [batch_size, input_len, self.n_heads, depth_size])
        output_heads = torch.reshape(x, output_shape)
        return torch.transpose(output_heads, 1, 2)

    def combine_heads(self, x):
        input_dims = list(torch._shape_as_tensor(x))
        batch_size = input_dims[0]
        input_len  = input_dims[2]
        num_heads  = input_dims[1]
        depth_size = input_dims[3]
        hidden_size = num_heads*depth_size
        
        output_shape  = tuple(
            [batch_size, input_len, hidden_size])
        output_tensor = torch.reshape(
            torch.transpose(x, 1, 2), output_shape)
        return output_tensor
    
    def forward(self, q, k, v, mask=None):
        q_input = self.split_heads(self.wq(q))
        k_input = self.split_heads(self.wk(k))
        v_input = self.split_heads(self.wv(v))
        
        attn_weights = self.dot_attention(
            q_input, k_input, mask=mask)
        attn_outputs = torch.matmul(
            attn_weights, v_input)
        attn_outputs = self.wc(
            self.combine_heads(attn_outputs))
        return attn_outputs, attn_weights

class FFWNetwork(torch.nn.Module):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = torch.nn.Linear(d_model, d_ffwd)
        self.ffwd_2 = torch.nn.Linear(d_ffwd, d_model)
    
    def forward(self, x):
        # Use the ReLU activation. #
        return self.ffwd_2(F.relu(self.ffwd_1(x)))

# Encoder Layer. #
class EncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        rate1=0.1, rate2=0.1, neg_infty=-1.0e9):
        super(EncoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        
        self.dropout_1 = torch.nn.Dropout(rate1)
        self.dropout_2 = torch.nn.Dropout(rate2)
    
    def forward(
        self, x_enc, x_pos, training=True):
        x_embed = x_enc + x_pos
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=None)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = torch.add(
            x_embed, self.lnorm_1(attn_self_tuple[0]))
        if training:
            attn_self_output = self.dropout_1(attn_self_output)
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = torch.add(
            attn_self_output, ffwd_self_output)
        if training:
            ffwd_self_output = self.dropout_2(ffwd_self_output)
        return ffwd_self_output

# Encoder. #
class Encoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Encoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = torch.rsqrt(
            torch.as_tensor(d_model, dtype=torch.float32))
        self.vocab_size = vocab_size

        # Embedding Layers. #
        self.pos_embed = torch.nn.ModuleList()
        self.voc_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # Encoder Layers. #
        self.layers = torch.nn.ModuleList()
        for m in range(n_layers):
            self.pos_embed.append(
                torch.nn.Embedding(max_seq_length, d_model))
            self.layers.append(EncoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        self.emb_dropout = torch.nn.Dropout(rate1)
    
    def forward(self, x, training=True):
        seq_length  = list(torch._shape_as_tensor(x))[1]
        x_pos_index = torch.unsqueeze(torch.arange(
            0, seq_length, dtype=torch.long), 0)
        if torch.cuda.is_available():
            x_pos_index = x_pos_index.to("cuda")

        x_tok_embed = self.voc_embed(x)
        x_tok_embed = x_tok_embed * self.d_rsqrt
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = x_pos_embed * self.d_rsqrt
            if training:
                x_pos_embed = self.emb_dropout(x_pos_embed)
            
            layer_output = self.layers[m](
                layer_input, x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

# Decoder Layer. #
class DecoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        rate1=0.1, rate2=0.1, neg_infty=-1.0e9):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.ffwd_cross = FFWNetwork(d_ffwd, d_model)
        self.attn_cross = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_3 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_4 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        
        self.dropout_1 = torch.nn.Dropout(rate1)
        self.dropout_2 = torch.nn.Dropout(rate2)
    
    def forward(
        self, x_enc, x_dec, 
        x_pos, mask, training=True):
        x_embed = x_dec + x_pos
        
        # Decoder Self-Attention. #
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=mask)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = torch.add(
            x_embed, self.lnorm_1(attn_self_tuple[0]))
        if training:
            attn_self_output = self.dropout_1(attn_self_output)
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = torch.add(
            attn_self_output, ffwd_self_output)
        if training:
            ffwd_self_output = self.dropout_2(ffwd_self_output)
        
        # Encoder Decoder Attention. #
        attn_cross_tuple  = self.attn_cross(
            ffwd_self_output, x_enc, x_enc, mask=None)
        attn_cross_output = torch.add(
            ffwd_self_output, self.lnorm_3(attn_cross_tuple[0]))
        if training:
            attn_cross_output = self.dropout_1(attn_cross_output)
        
        ffwd_cross_output = self.lnorm_4(
            self.ffwd_cross(attn_cross_output))
        ffwd_cross_output = torch.add(
            attn_cross_output, ffwd_cross_output)
        if training:
            ffwd_cross_output = self.dropout_2(ffwd_cross_output)
        return ffwd_cross_output

# Decoder. #
class Decoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = torch.rsqrt(
            torch.as_tensor(d_model, dtype=torch.float32))
        self.vocab_size = vocab_size

        # Embedding Layers. #
        self.pos_embed = torch.nn.ModuleList()
        self.voc_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # Transformer Layers. #
        self.layers = torch.nn.ModuleList()
        for m in range(n_layers):
            self.pos_embed.append(
                torch.nn.Embedding(max_seq_length, d_model))
            self.layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        self.emb_dropout = torch.nn.Dropout(rate1)
    
    def forward(self, x_enc, x_dec, training=True):
        dec_length = list(torch._shape_as_tensor(x_dec))[1]
        input_zero = torch.zeros(dec_length, dec_length)
        input_mask = torch.triu(1.0 + input_zero, diagonal=1)
        input_mask = torch.unsqueeze(
            torch.unsqueeze(input_mask, 0), 0)
        
        x_pos_index = torch.unsqueeze(torch.arange(
            0, dec_length, dtype=torch.long), 0)
        if torch.cuda.is_available():
            x_pos_index = x_pos_index.to("cuda")
        
        x_tok_embed = self.voc_embed(x_dec)
        x_tok_embed = x_tok_embed * self.d_rsqrt
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = x_pos_embed * self.d_rsqrt
            if training:
                x_pos_embed = self.emb_dropout(x_pos_embed)
            
            layer_output = self.layers[m](
                x_enc, layer_input, x_pos_embed, 
                input_mask, training=training)
            layer_input  = layer_output
        return layer_output

class Transformer(torch.nn.Module):
    def __init__(
        self, n_layers, n_heads, d_model, 
        d_ffwd, enc_vocab_size, dec_vocab_size, 
        enc_seq_len, dec_seq_len, rate1=0.1, rate2=0.1):
        super(Transformer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len

        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        
        # Define the encoder and decoder. #
        self.encoder = Encoder(
            n_layers, d_model, 
            n_heads, d_ffwd, enc_vocab_size, 
            enc_seq_len, rate1=rate1, rate2=rate2)
        
        self.decoder = Decoder(
            n_layers, d_model, 
            n_heads, d_ffwd, dec_vocab_size, 
            dec_seq_len, rate1=rate1, rate2=rate2)
        
        # Define the output projection. #
        self.projection = torch.nn.Linear(
            d_model, dec_vocab_size, bias=False)
    
    def forward(self, x_enc, x_dec, training=True):
        enc_outputs = self.encoder(
            x_enc, training=training)
        
        dec_outputs = self.decoder(
            enc_outputs, x_dec, training=training)
        dec_logits  = self.projection(dec_outputs)
        return dec_logits
    
    def infer(self, x_enc, x_dec, k=1):
        input_len = list(torch._shape_as_tensor(x_dec))[1]
        infer_ids = [torch.unsqueeze(x_dec[:, 0], 1)]
        
        for step in range(self.dec_seq_len):
            tmp_inputs = torch.cat(infer_ids, dim=1)
            with torch.no_grad():
                tmp_logits = self.forward(
                    x_enc, tmp_inputs, training=False)

                if step < (input_len-1):
                    tmp_argmax = x_dec[:, step+1]
                    infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                else:
                    if k == 1:
                        tmp_argmax =  torch.argmax(
                            tmp_logits[:, -1, :], dim=1)
                        infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                    else:
                        tmp_logit = tmp_logits[:, -1, :]
                        tmp_prob  = F.softmax(tmp_logit, dim=1)

                        tmp_top_k  = torch.topk(tmp_prob, k=k)
                        tmp_sample = torch.multinomial(
                            tmp_top_k.values, 1)
                        tmp_index  = torch.gather(
                            tmp_top_k.indices, 1, tmp_sample)
                        infer_ids.append(tmp_index)
        return torch.cat(infer_ids, dim=1)
