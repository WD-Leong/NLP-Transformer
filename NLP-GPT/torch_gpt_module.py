
import torch

def split_heads(x, num_heads):
    input_dims = list(x.size())
    batch_size = input_dims[0]
    input_len  = input_dims[1]
    depth_size = int(input_dims[2] / num_heads)
    
    output_shape = tuple(
        [batch_size, input_len, num_heads, depth_size])
    output_heads = torch.reshape(x, output_shape)
    return torch.transpose(output_heads, 1, 2)

def combine_heads(x):
    input_dims = list(x.size())
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

def layer_normalisation(x, bias, scale, eps=1.0e-6):
    x_mean = torch.mean(x, 2, True)
    x_var  = torch.mean(
        torch.square(x - x_mean), 2, True)
    x_norm = (x - x_mean) * torch.rsqrt(x_var + eps)
    return (x_norm * scale) + bias

class GPT_Network(torch.nn.Module):
    def __init__(
    self, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length, embed_size=128, p_keep=0.9, p_reg=1.0, 
    var_type="norm_add", attn_type="mult_attn", **kwargs):
        super(GPT_Network, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type  = var_type
        self.attn_type = attn_type
        self.seq_length  = seq_length
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.ffwd_size   = ffwd_size
        self.head_size   = int(hidden_size / n_heads)
        
        # Embedding matrices. #
        self.emb_decode = torch.nn.Embedding(
            self.vocab_size, self.embed_size)
        self.W_dec_lin  = torch.nn.Parameter(
            0.1 * torch.randn(self.embed_size, self.hidden_size))
        
        # Output projection. #
        self.p_decoder = torch.nn.Parameter(
            0.1 * torch.randn(self.hidden_size, self.vocab_size))
        
        # Position Embeddings. #
        self.x_emb_pos_dec = \
            torch.nn.Parameter(0.1 * torch.randn(
                self.n_layers, self.seq_length, self.hidden_size))
        
        # GPT Variables. #
        self.p_d_q = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_k = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_v = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_c = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        
        if self.attn_type == "add_attn":
            self.v_d_s = torch.nn.Parameter(
                0.1 * torch.randn(self.n_layers, self.head_size, 1))
        
        self.p_d_ff1 = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.ffwd_size))
        self.p_d_ff2 = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.ffwd_size, self.hidden_size))
        self.b_d_ff1 = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.ffwd_size))
        self.b_d_ff2 = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        
        self.d_o_bias  = torch.nn.Parameter(
            torch.zeros(self.hidden_size))
        self.d_o_scale = torch.nn.Parameter(
            1.0 + torch.zeros(self.hidden_size))
        
        self.b_d_bias_1  = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_bias_2  = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_scale_1 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_scale_2 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
    
    def forward(self, x, p_keep=1.0, p_reg=1.0):
        step = list(x.size())[1]
        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=3)
        dropout = torch.nn.Dropout(1.0-p_reg)
        dropout_ffw = torch.nn.Dropout(1.0-p_keep)
        
        head_size = torch.tensor(
            self.head_size, dtype=torch.float32)
        neg_infty = -1.0e9
        attn_mask = neg_infty * torch.triu(
            1.0 + torch.zeros(step, step), diagonal=1)
        attn_mask = torch.unsqueeze(
            torch.unsqueeze(attn_mask, 0), 0)
        if torch.cuda.is_available():
            attn_mask = attn_mask.cuda()
        
        x_dec_token = self.emb_decode(x)
        x_dec_embed = torch.matmul(
            x_dec_token, self.W_dec_lin)
        layer_input = x_dec_embed
        for m in range(self.n_layers):
            layer_in = \
                self.x_emb_pos_dec[m, :step, :] + layer_input
            
            # Self Attention Layer. #
            x_sq = split_heads(torch.matmul(
                layer_in, self.p_d_q[m]), self.n_heads)
            x_sq = x_sq * torch.rsqrt(head_size)
            
            x_sk = split_heads(torch.matmul(
                layer_in, self.p_d_k[m]), self.n_heads)
            x_sv = split_heads(torch.matmul(
                layer_in, self.p_d_v[m]), self.n_heads)
            
            if self.attn_type == "add_attn":
                x_sq = torch.unsqueeze(x_sq, axis=3)
                x_sk = torch.unsqueeze(x_sk, axis=2)
                
                x_s_scores = torch.matmul(
                    tanh(x_sq + x_sk), self.v_d_s[m])
                x_s_alphas = softmax(
                    torch.squeeze(x_s_scores, 4) + attn_mask)
            else:
                x_s_scores = \
                    torch.matmul(x_sq, torch.transpose(x_sk, 2, 3))
                x_s_alphas = softmax(x_s_scores + attn_mask)
            
            x_self_conc  = \
                combine_heads(torch.matmul(x_s_alphas, x_sv))
            x_multi_self = dropout(
                torch.matmul(x_self_conc, self.p_d_c[m]))
            
            if self.var_type == "norm_add":
                x_self_norm = layer_in + layer_normalisation(
                    x_multi_self, self.b_d_bias_1[m], self.b_d_scale_1[m])
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(
                    layer_in + x_multi_self, 
                    self.b_d_bias_1[m], self.b_d_scale_1[m])
            
            # Feed forward layer. #
            x_ffw1 = relu(self.b_d_ff1[m] +\
                torch.matmul(x_self_norm, self.p_d_ff1[m]))
            x_ffw2 = self.b_d_ff2[m] +\
                torch.matmul(x_ffw1, self.p_d_ff2[m])
            x_ffwd = dropout_ffw(x_ffw2)
            
            if self.var_type == "norm_add":
                x_ffw_norm = x_self_norm + layer_normalisation(
                    x_ffwd, self.b_d_bias_2[m], self.b_d_scale_2[m])
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    x_self_norm + x_ffwd, 
                    self.b_d_bias_2[m], self.b_d_scale_2[m])
            
            # Append the output. #
            layer_in = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            dec_outputs = x_dec_embed + layer_normalisation(
                x_ffw_norm, self.d_o_bias, self.d_o_scale)
        elif self.var_type == "add_norm":
            dec_outputs = layer_normalisation(
                x_dec_embed + x_ffw_norm, 
                self.d_o_bias, self.d_o_scale)
        dec_logits = torch.matmul(dec_outputs, self.p_decoder)
        return dec_logits

def infer(model, x, inf_length):
    # Inference. #
    infer_len = list(x.size())[1]
    infer_ids = [torch.unsqueeze(x[:, 0], axis=1)]
    
    for step in range(inf_length):
        x_input = torch.cat(tuple(infer_ids), dim=1)
        
        tmp_logits = model(x_input, p_keep=1.0)
        if step < (infer_len-1):
            tmp_argmax = x[:, step+1]
        else:
            tmp_argmax =  torch.argmax(
                tmp_logits[:, -1, :], axis=1)
        infer_ids.append(torch.unsqueeze(tmp_argmax, axis=1))
    
    infer_indices = torch.cat(tuple(infer_ids), dim=1)
    return infer_indices
