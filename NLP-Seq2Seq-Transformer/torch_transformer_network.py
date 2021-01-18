
import torch

# Custom function to initialise the weights. #
def init_weights(m):
    torch.nn.init.normal_(m.weight, std=0.1)

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

class TransformerNetwork(torch.nn.Module):
    def __init__(
        self, n_layers, n_heads, 
        hidden_size, ffwd_size, 
        enc_vocab_size, dec_vocab_size, 
        seq_enc, seq_dec, embed_size=128, 
        var_type="norm_add", attn_type="mult_attn", **kwargs):
        super(TransformerNetwork, self).__init__(**kwargs)
        
        self.n_heads  = n_heads
        self.var_type = var_type
        self.n_layers = n_layers
        self.attn_type  = attn_type
        self.seq_encode = seq_enc
        self.seq_decode = seq_dec
        self.hidden_size = hidden_size
        self.head_size   = int(hidden_size / n_heads)
        self.ffwd_size   = ffwd_size
        self.embed_size  = embed_size
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        
        self.emb_encode = torch.nn.Embedding(
            self.enc_vocab_size, self.embed_size)
        self.W_enc_lin  = torch.nn.Parameter(
            0.1 * torch.randn(self.embed_size, self.hidden_size))
        self.emb_decode = torch.nn.Embedding(
            self.dec_vocab_size, self.embed_size)
        self.W_dec_lin  = torch.nn.Parameter(
            0.1 * torch.randn(self.embed_size, self.hidden_size))
        
        self.emb_encode.apply(init_weights)
        self.emb_decode.apply(init_weights)
        
        # Output projection. #
        self.p_decoder = \
            torch.nn.Parameter(0.1 * torch.randn(
                self.hidden_size, self.dec_vocab_size))
        
        # Transformer Encoder Variables. #
        self.e_pos_embed = \
            torch.nn.Parameter(0.1 * torch.randn(
                self.n_layers, self.seq_encode, self.hidden_size))
        self.p_e_q = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_e_k = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_e_v = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_e_c = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        
        self.p_e_ff1 = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.ffwd_size))
        self.p_e_ff2 = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.ffwd_size, self.hidden_size))
        self.b_e_ff1 = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.ffwd_size))
        self.b_e_ff2 = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        
        self.e_o_bias  = torch.nn.Parameter(
            torch.zeros(self.hidden_size))
        self.e_o_scale = torch.nn.Parameter(
            1.0 + torch.zeros(self.hidden_size))
        
        self.b_e_bias_1  = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        self.b_e_bias_2  = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        self.b_e_scale_1 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
        self.b_e_scale_2 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
        
        # Transformer Decoder Variables. #
        self.d_pos_embed = \
            torch.nn.Parameter(0.1 * torch.randn(
                self.n_layers, self.seq_decode, self.hidden_size))
        self.p_d_q = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_k = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_v = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_d_c = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        
        self.p_a_q = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_a_k = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_a_v = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        self.p_a_c = torch.nn.Parameter(0.1 * torch.randn(
            self.n_layers, self.hidden_size, self.hidden_size))
        
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
        self.b_d_bias_3  = torch.nn.Parameter(
            torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_scale_1 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_scale_2 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
        self.b_d_scale_3 = torch.nn.Parameter(
            1.0 + torch.zeros(self.n_layers, self.hidden_size))
    
    def transformer_encode(self, x, p_reg=1.00, p_keep=0.90):
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=3)
        dropout = torch.nn.Dropout(1.0-p_reg)
        dropout_ffw = torch.nn.Dropout(1.0-p_keep)
        
        head_size = torch.tensor(
            self.head_size, dtype=torch.float32)
        
        x_enc_token = self.emb_encode(x)
        x_enc_embed = torch.matmul(
            x_enc_token, self.W_enc_lin)
        layer_input = x_enc_embed
        for m in range(self.n_layers):
            layer_in = self.e_pos_embed[m] + layer_input
            
            # Self Attention Layer. #
            x_e_q = split_heads(torch.matmul(
                layer_in, self.p_e_q[m]), self.n_heads)
            x_e_q = x_e_q * torch.rsqrt(head_size)
            
            x_e_k = split_heads(torch.matmul(
                layer_in, self.p_e_k[m]), self.n_heads)
            x_e_v = split_heads(torch.matmul(
                layer_in, self.p_e_v[m]), self.n_heads)
            
            x_scores = torch.matmul(
                x_e_q, torch.transpose(x_e_k, 2, 3))
            x_alphas = softmax(x_scores)
            
            x_head_conc  = \
                combine_heads(torch.matmul(x_alphas, x_e_v))
            x_multi_head = dropout(
                torch.matmul(x_head_conc, self.p_e_c[m]))
            
            # Residual Connection Layer. #
            if self.var_type == "norm_add":
                x_heads_norm = layer_in + layer_normalisation(
                    x_multi_head, self.b_e_bias_1[m], self.b_e_scale_1[m])
            elif self.var_type == "add_norm":
                x_heads_norm = layer_normalisation(
                    layer_in + x_multi_head, 
                    self.b_e_bias_1[m], self.b_e_scale_1[m])
            
            # Feed forward layer. #
            x_ffw1 = relu(self.b_e_ff1[m] +\
                torch.matmul(x_heads_norm, self.p_e_ff1[m]))
            x_ffw2 = self.b_e_ff2[m] +\
                torch.matmul(x_ffw1, self.p_e_ff2[m])
            x_ffwd = dropout_ffw(x_ffw2)
            
            # Residual Connection Layer. #
            if self.var_type == "norm_add":
                x_ffw_norm = x_heads_norm + layer_normalisation(
                    x_ffwd, self.b_e_bias_2[m], self.b_e_scale_2[m])
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    x_heads_norm + x_ffwd, 
                    self.b_e_bias_2[m], self.b_e_scale_2[m])
            
            # Append the output of the current layer. #
            layer_input = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            enc_outputs = \
                x_enc_embed + layer_normalisation(
                    x_ffw_norm, self.e_o_bias, self.e_o_scale)
        elif self.var_type == "add_norm":
            enc_outputs = layer_normalisation(
                x_enc_embed + x_ffw_norm, 
                self.e_o_bias, self.e_o_scale)
        return enc_outputs
    
    def transformer_decode(
        self, dec_inputs, enc_outputs, p_reg=1.00, p_keep=0.90):
        step = list(dec_inputs.size())[1]
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
        
        x_dec_token = self.emb_decode(dec_inputs)
        x_dec_embed = torch.matmul(
            x_dec_token, self.W_dec_lin)
        layer_input = x_dec_embed
        for m in range(self.n_layers):
            layer_in = self.d_pos_embed[m, :step, :] + layer_input
            
            # Masked Self Attention Layer. #
            x_d_q = split_heads(torch.matmul(
                layer_in, self.p_d_q[m]), self.n_heads)
            x_d_q = x_d_q * torch.rsqrt(head_size)
            
            x_d_k = split_heads(torch.matmul(
                layer_in, self.p_d_k[m]), self.n_heads)
            x_d_v = split_heads(torch.matmul(
                layer_in, self.p_d_v[m]), self.n_heads)
            
            x_self_scores = torch.matmul(
                x_d_q, torch.transpose(x_d_k, 2, 3))
            x_self_alphas = \
                softmax(x_self_scores + attn_mask)
            
            x_self_conc = \
                combine_heads(torch.matmul(x_self_alphas, x_d_v))
            x_self_head = dropout(
                torch.matmul(x_self_conc, self.p_d_c[m]))
            
            if self.var_type == "norm_add":
                x_self_norm = layer_in + layer_normalisation(
                    x_self_head, self.b_d_bias_1[m], self.b_d_scale_1[m])
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(
                    layer_in + x_self_head, 
                    self.b_d_bias_1[m], self.b_d_scale_1[m])
            
            # Encoder Decoder Attention Layer. #
            x_a_q = split_heads(torch.matmul(
                x_self_norm, self.p_a_q[m]), self.n_heads)
            x_a_q = x_a_q * torch.rsqrt(head_size)
            
            x_a_k = split_heads(torch.matmul(
                enc_outputs, self.p_a_k[m]), self.n_heads)
            x_a_v = split_heads(torch.matmul(
                enc_outputs, self.p_a_v[m]), self.n_heads)
            
            x_attn_scores = torch.matmul(
                x_a_q, torch.transpose(x_a_k, 2, 3))
            x_attn_alphas = softmax(x_attn_scores)
            
            x_head_conc  = \
                combine_heads(torch.matmul(x_attn_alphas, x_a_v))
            x_multi_head = dropout(
                torch.matmul(x_head_conc, self.p_a_c[m]))
            
            if self.var_type == "norm_add":
                x_heads_norm = x_self_norm + layer_normalisation(
                    x_multi_head, self.b_d_bias_2[m], self.b_d_scale_2[m])
            elif self.var_type == "add_norm":
                x_heads_norm = layer_normalisation(
                    x_self_norm + x_multi_head, 
                    self.b_d_bias_2[m], self.b_d_scale_2[m])
            
            # Feed forward layer. #
            x_ffw1 = relu(self.b_d_ff1[m] +\
                torch.matmul(x_heads_norm, self.p_d_ff1[m]))
            x_ffw2 = self.b_d_ff2[m] +\
                torch.matmul(x_ffw1, self.p_d_ff2[m])
            x_ffwd = dropout_ffw(x_ffw2)
            
            # Residual Connection Layer. #
            if self.var_type == "norm_add":
                x_ffw_norm = x_heads_norm + layer_normalisation(
                    x_ffwd, self.b_d_bias_3[m], self.b_d_scale_3[m])
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    x_heads_norm + x_ffwd, 
                    self.b_d_bias_3[m], self.b_d_scale_3[m])
            
            # Append the output of the current layer. #
            layer_input = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            dec_outputs = \
                x_dec_embed + layer_normalisation(
                    x_ffw_norm, self.d_o_bias, self.d_o_scale)
        elif self.var_type == "add_norm":
            dec_outputs = layer_normalisation(
                x_dec_embed + x_ffw_norm, self.d_o_bias, self.d_o_scale)
        return dec_outputs
    
    def forward(
        self, x_encode, x_decode, 
        p_reg=1.0, p_keep=0.9, infer=False):
        # Encoder. #
        enc_outputs = self.transformer_encode(
            x_encode, p_reg=p_reg, p_keep=p_keep)
        
        # Decoder Model. #
        if infer:
            # Inference. #
            infer_index = [torch.unsqueeze(x_decode[:, 0], axis=1)]
            for step in range(self.seq_decode):
                x_infer_ids = torch.cat(tuple(infer_index), dim=1)
                tmp_outputs = self.transformer_decode(
                    x_infer_ids, enc_outputs, p_reg=1.0, p_keep=1.0)
                
                tmp_logit  = torch.matmul(
                    tmp_outputs[:, -1, :], self.p_decoder)
                tmp_argmax = torch.argmax(tmp_logit, axis=1)
                infer_index.append(torch.unsqueeze(tmp_argmax, axis=1))
            
            # Concatenate the list into a tensor. #
            infer_index = torch.cat(tuple(infer_index), axis=1)
            return infer_index[:, 1:]
        else:
            dec_outputs = self.transformer_decode(
                x_decode, enc_outputs, p_reg=p_reg, p_keep=p_keep)
            dec_logits  = \
                torch.matmul(dec_outputs, self.p_decoder)
            return dec_logits

