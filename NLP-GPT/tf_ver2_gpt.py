
import tensorflow as tf

def split_heads(x, num_heads):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[1]
    depth_size = tf.cast(
        tf.shape(x)[2] / num_heads, tf.int32)

    split_outputs = tf.reshape(
        x, [batch_size, input_len, num_heads, depth_size])
    return tf.transpose(split_outputs, [0, 2, 1, 3])

def combine_heads(x):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[2]
    num_heads  = tf.shape(x)[1]
    depth_size = tf.shape(x)[3]
    hidden_size = num_heads*depth_size

    combined_outputs = tf.reshape(tf.transpose(
        x, [0, 2, 1, 3]), [batch_size, input_len, hidden_size])
    return combined_outputs

def layer_normalisation(x, bias, scale, eps=1.0e-6):
    x_mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    x_var  = tf.reduce_mean(
        tf.square(x - x_mean), axis=[-1], keepdims=True)
    x_norm = (x - x_mean) * tf.math.rsqrt(x_var + tf.constant(eps))
    return (x_norm * scale) + bias

def transformer_decode(
    step, n_layers, n_heads, pos_embed, 
    dec_inputs, b_sq, b_sk, p_sq, p_sk, p_sv, p_sc, 
    p_ff1, p_ff2, b_ff1, b_ff2, i_bias, i_scale, 
    b_bias_1, b_scale_1, b_bias_2, b_scale_2, o_bias, o_scale, 
    p_keep=0.90, p_reg=1.00, var_type="add_norm", eps=1.0e-8):
    hidden_sz = tf.shape(dec_inputs)[2]
    head_size = tf.cast(hidden_sz / n_heads, tf.float32)

    neg_infty = -1.0e9
    attn_mask = neg_infty *\
        (1.0 - tf.linalg.band_part(tf.ones([step, step]), -1, 0))
    attn_mask = tf.expand_dims(
        tf.expand_dims(attn_mask, axis=0), axis=0)
    
    layer_in  = dec_inputs
    for m in range(n_layers):
        if var_type == "norm_add":
            layer_in = pos_embed[m, :step, :] +\
                layer_normalisation(layer_in, i_bias[m], i_scale[m])
        elif var_type == "add_norm":
            layer_in = layer_normalisation(
                pos_embed[m, :step, :] + layer_in, i_bias[m], i_scale[m])
        
        # Self Attention Layer. #
        x_sq = split_heads(b_sq[m, :step, :] +\
            tf.tensordot(layer_in, p_sq[m], [[2], [0]]), n_heads)
        x_sq = x_sq * tf.math.rsqrt(head_size)

        x_sk = split_heads(b_sk[m, :step, :] +\
            tf.tensordot(layer_in, p_sk[m], [[2], [0]]), n_heads)
        x_sv = split_heads(tf.tensordot(
            layer_in, p_sv[m], [[2], [0]]), n_heads)
        
        x_s_scores = tf.matmul(
            x_sq, x_sk, transpose_b=True)
        x_s_alphas = tf.nn.softmax(x_s_scores + attn_mask)
        
        x_self_conc  = combine_heads(tf.matmul(x_s_alphas, x_sv))
        x_multi_self = tf.nn.dropout(tf.tensordot(
            x_self_conc, p_sc[m], [[2], [0]]), rate=1.0-p_reg)
        
        if var_type == "norm_add":
            x_self_norm = tf.add(
                layer_in, layer_normalisation(
                    x_multi_self, b_bias_1[m], b_scale_1[m]))
        elif var_type == "add_norm":
            x_self_norm = layer_normalisation(
                layer_in + x_multi_self, b_bias_1[m], b_scale_1[m])

        # Feed forward layer. #
        x_ffw1 = tf.nn.relu(tf.tensordot(
            x_self_norm, p_ff1[m], [[2], [0]]) + b_ff1[m])
        x_ffw2 = b_ff2[m] + tf.tensordot(x_ffw1, p_ff2[m], [[2], [0]])
        x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)

        if var_type == "norm_add":
            x_ffw_norm = tf.add(
                x_self_norm, layer_normalisation(
                    x_ffwd, b_bias_2[m], b_scale_2[m]))
        elif var_type == "add_norm":
            x_ffw_norm = layer_normalisation(
                x_self_norm + x_ffwd, b_bias_2[m], b_scale_2[m])
        
        # Append the output. #
        layer_in = x_ffw_norm
    
    if var_type == "norm_add":
        dec_outputs = dec_inputs +\
            layer_normalisation(x_ffw_norm, o_bias, o_scale)
    elif var_type == "add_norm":
        dec_outputs = layer_normalisation(
            dec_inputs + x_ffw_norm, o_bias, o_scale)
    return dec_outputs

class GPT_Network(tf.keras.Model):
    def __init__(
    self, n_layers, n_heads, hidden_size, ffwd_size, 
    vocab_size, seq_length, embed_size=128, 
    p_keep=0.9, p_reg=1.0, var_type="norm_add", **kwargs):
        super(GPT_Network, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type = var_type
        self.seq_length  = seq_length
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.ffwd_size   = ffwd_size
        self.head_size   = int(hidden_size / n_heads)
        
        # Embedding matrices. #
        self.W_emb_dec = tf.Variable(tf.random.normal(
            [self.vocab_size, self.embed_size], stddev=0.1), name="dec_embed")
        self.W_dec_lin = tf.Variable(tf.random.normal(
            [self.embed_size, self.hidden_size], stddev=0.1), name="dec_lin")
        
        # Output projection. #
        self.p_decoder = tf.Variable(tf.random.normal(
            [self.hidden_size, self.vocab_size], stddev=0.1), name="p_decoder")
        
        # GPT Variables. #
        attn_add_shape  = [self.n_layers, self.head_size, 1]
        attn_bias_shape = [self.n_layers, self.seq_length, self.hidden_size]
        attn_wgt_shape  = [self.n_layers, self.hidden_size, self.hidden_size]
        attn_ffw1_shape = [self.n_layers, self.hidden_size, self.ffwd_size]
        attn_ffw2_shape = [self.n_layers, self.ffwd_size, self.hidden_size]
        pos_embed_shape = [self.n_layers, self.seq_length, self.hidden_size]
        
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_c")
        
        self.b_d_q = tf.Variable(
            tf.zeros(attn_bias_shape), name="b_d_q")
        self.b_d_k = tf.Variable(
            tf.zeros(attn_bias_shape), name="b_d_k")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
        self.d_i_bias  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="d_i_bias")
        self.d_i_scale = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="d_i_scale")
        self.d_o_bias  = tf.Variable(tf.zeros(
            [self.hidden_size]), name="d_o_bias")
        self.d_o_scale = tf.Variable(tf.ones(
            [self.hidden_size]), name="d_o_scale")
        
        self.b_d_bias_1  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_d_bias_1")
        self.b_d_bias_2  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_d_bias_2")
        self.b_d_scale_1 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_d_scale_1")
        self.b_d_scale_2 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_d_scale_2")
        
        # Position Embeddings. #
        self.x_emb_pos_dec = tf.Variable(tf.random.normal(
            pos_embed_shape, stddev=0.1), name="pos_embed")
    
    def call(self, x_input):
        # Word or Sub-word embeddings. #
        x_dec_token = tf.nn.embedding_lookup(self.W_emb_dec, x_input)
        
        # Transformer Decoder. #
        x_dec_embed = tf.tensordot(
            x_dec_token, self.W_dec_lin, [[2], [0]])
        
        # Training via Teacher forcing. #
        dec_outputs = transformer_decode(
            self.seq_length, self.n_layers, 
            self.n_heads, self.x_emb_pos_dec, x_dec_embed, 
            self.b_d_q, self.b_d_k, self.p_d_q, self.p_d_k, 
            self.p_d_v, self.p_d_c, self.p_d_ff1, self.p_d_ff2, 
            self.b_d_ff1, self.b_d_ff2, self.d_i_bias, 
            self.d_i_scale, self.b_d_bias_1, self.b_d_scale_1, 
            self.b_d_bias_2, self.b_d_scale_2, self.d_o_bias, 
            self.d_o_scale, p_keep=self.p_keep, var_type=self.var_type)
        
        dec_logits = tf.tensordot(
            dec_outputs, self.p_decoder, [[2], [0]])
        return dec_logits
    
    def infer(self, x_infer):
        # Inference. #
        infer_len = tf.shape(x_infer)[1]
        x_inf_emb = tf.nn.embedding_lookup(self.W_emb_dec, x_infer)
        infer_emb = [tf.expand_dims(x_inf_emb[:, 0, :], axis=1)]
        infer_ids = [tf.expand_dims(x_infer[:, 0], axis=1)]
        
        for step in range(self.seq_length):
            x_inf_inputs = tf.concat(infer_emb, axis=1)
            x_inf_inputs = tf.tensordot(
                x_inf_inputs, self.W_dec_lin, [[2], [0]])
            
            tmp_outputs = transformer_decode(
                step+1, self.n_layers, self.n_heads, 
                self.x_emb_pos_dec, x_inf_inputs, 
                self.b_d_q, self.b_d_k, self.p_d_q, self.p_d_k, 
                self.p_d_v, self.p_d_c, self.p_d_ff1, self.p_d_ff2, 
                self.b_d_ff1, self.b_d_ff2, self.d_i_bias, 
                self.d_i_scale, self.b_d_bias_1, self.b_d_scale_1, 
                self.b_d_bias_2, self.b_d_scale_2, self.d_o_bias, 
                self.d_o_scale, p_keep=1.0, var_type=self.var_type)
            
            tmp_logit  = tf.matmul(
                tmp_outputs[:, -1, :], self.p_decoder)
            tmp_argmax = tf.cond(
                step < (infer_len-1), 
                lambda: x_infer[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            next_embed = tf.cond(
                step < (infer_len-1), 
                lambda: x_inf_emb[:, step+1, :], 
                lambda: tf.matmul(
                    tf.nn.softmax(tmp_logit), self.W_emb_dec))
            
            infer_ids.append(tf.expand_dims(tmp_argmax, axis=1))
            infer_emb.append(tf.expand_dims(next_embed, axis=1))
        
        infer_indices = tf.concat(infer_ids, axis=1)
        return infer_indices
