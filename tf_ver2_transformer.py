# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:56:21 2019

@author: User
"""

import tensorflow as tf

def split_heads(full_inputs, num_heads):
    batch_size = tf.shape(full_inputs)[0]
    input_len  = tf.shape(full_inputs)[1]
    depth_size = tf.shape(full_inputs)[2] // num_heads
    
    split_outputs = tf.reshape(
        full_inputs, [batch_size, input_len, num_heads, depth_size])
    return tf.transpose(split_outputs, [0, 2, 1, 3])

def combine_heads(head_inputs):
    batch_size = tf.shape(head_inputs)[0]
    seq_length = tf.shape(head_inputs)[2]
    num_heads  = tf.shape(head_inputs)[1]
    depth_size = tf.shape(head_inputs)[3]
    hidden_size = num_heads*depth_size
    
    combined_outputs = tf.reshape(tf.transpose(
        head_inputs, [0, 2, 1, 3]), [batch_size, seq_length, hidden_size])
    return combined_outputs

def layer_normalisation(layer_input, bias, scale, eps=1.0e-6):
    eps = tf.constant(eps)
    layer_mean = tf.reduce_mean(
        layer_input, axis=[-1], keepdims=True)
    layer_var  = tf.reduce_mean(
        tf.square(layer_input-layer_mean), axis=[-1], keepdims=True)
    
    layer_std = (layer_input-layer_mean) * tf.math.rsqrt(layer_var + eps)
    return (layer_std*scale) + bias

def transformer_encode(
    n_layers, n_heads, pos_embed, enc_inputs, 
    b_e_q, b_e_k, p_e_q, p_e_k, p_e_v, p_e_c, 
    p_ff1, b_ff1, p_ff2, b_ff2, i_bias, i_scale, 
    b_bias_1, b_scale_1, b_bias_2, b_scale_2, 
    o_bias, o_scale, p_keep=0.9, p_reg=1.00, var_type="norm_add"):
    head_size = tf.cast(
        tf.shape(enc_inputs)[2] / n_heads, tf.float32)
    
    layer_in = enc_inputs
    for m in range(n_layers):
        if var_type == "norm_add":
            layer_in = pos_embed[m] +\
                layer_normalisation(layer_in, i_bias[m], i_scale[m])
        elif var_type == "add_norm":
            layer_in = layer_normalisation(
                layer_in + pos_embed[m], i_bias[m], i_scale[m])
        
        # Self Attention Layer. #
        x_e_q = split_heads(tf.tensordot(
            layer_in, p_e_q[m], [[2], [0]]) + b_e_q[m], n_heads)
        x_e_q = x_e_q * tf.math.rsqrt(head_size)
        
        x_e_k = split_heads(tf.tensordot(
            layer_in, p_e_k[m], [[2], [0]]) + b_e_k[m], n_heads)
        x_e_v = split_heads(tf.tensordot(
            layer_in, p_e_v[m], [[2], [0]]), n_heads)
        
        x_scores = tf.matmul(
            x_e_q, x_e_k, transpose_b=True)
        x_alphas = tf.nn.softmax(x_scores)
        
        x_head_conc  = combine_heads(tf.matmul(x_alphas, x_e_v))
        x_multi_head = tf.nn.dropout(tf.tensordot(
            x_head_conc, p_e_c[m], [[2], [0]]), rate=1.0-p_reg)
        
        # Residual Connection Layer. #
        if var_type == "norm_add":
            x_heads_norm = tf.add(
                layer_in, layer_normalisation(
                    x_multi_head, b_bias_1[m], b_scale_1[m]))
        elif var_type == "add_norm":
            x_heads_norm = layer_normalisation(
                layer_in + x_multi_head, b_bias_1[m], b_scale_1[m])
        
        # Feed forward layer. #
        x_ffw1 = tf.nn.relu(tf.tensordot(
            x_heads_norm, p_ff1[m], [[2], [0]]) + b_ff1[m])
        x_ffw2 = tf.tensordot(x_ffw1, p_ff2[m], [[2], [0]]) + b_ff2[m]
        x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
        
        # Residual Connection Layer. #
        if var_type == "norm_add":
            x_ffw_norm = x_heads_norm +\
                layer_normalisation(x_ffwd, b_bias_2[m], b_scale_2[m])
        elif var_type == "add_norm":
            x_ffw_norm = layer_normalisation(
                x_heads_norm + x_ffwd, b_bias_2[m], b_scale_2[m])
        
        # Append the output of the current layer. #
        layer_in = x_ffw_norm
    
    if var_type == "norm_add":
        enc_outputs = enc_inputs +\
            layer_normalisation(x_ffw_norm, o_bias, o_scale)
    elif var_type == "add_norm":
        enc_outputs = layer_normalisation(
            enc_inputs + x_ffw_norm, o_bias, o_scale)
    return enc_outputs

def transformer_decode(
    step, n_layers, n_heads, 
    pos_embed, dec_inputs, enc_outputs, 
    b_d_q, b_d_k, b_a_q, b_a_k, p_d_q, p_d_k, 
    p_d_v, p_d_c, p_a_q, p_a_k, p_a_v, p_a_c, 
    p_ff1, b_ff1, p_ff2, b_ff2, i_bias, i_scale, 
    b_bias_1, b_scale_1, b_bias_2, b_scale_2, b_bias_3, b_scale_3, 
    o_bias, o_scale, p_keep=0.9, p_reg=1.00, var_type="norm_add"):
    head_size = tf.cast(
        tf.shape(enc_outputs)[2] / n_heads, tf.float32)
    
    neg_infty = -1.0e9
    attn_mask = neg_infty *\
        (1.0 - tf.linalg.band_part(tf.ones([step, step]), -1, 0))
    attn_mask = tf.expand_dims(
        tf.expand_dims(attn_mask, axis=0), axis=0)
    
    layer_in = dec_inputs
    for m in range(n_layers):
        if var_type == "norm_add":
            layer_in = pos_embed[m, :step, :] +\
                layer_normalisation(layer_in, i_bias[m], i_scale[m])
        elif var_type == "add_norm":
            layer_in = layer_normalisation(
                layer_in + pos_embed[m, :step, :], i_bias[m], i_scale[m])
        
        # Masked Self Attention Layer. #
        x_d_q = split_heads(tf.add(
            b_d_q[m, :step, :], tf.tensordot(
                layer_in, p_d_q[m], [[2], [0]])), n_heads)
        x_d_q = x_d_q * tf.math.rsqrt(head_size)
        
        x_d_k = split_heads(tf.add(
            b_d_k[m, :step, :], tf.tensordot(
                layer_in, p_d_k[m], [[2], [0]])), n_heads)
        x_d_v = split_heads(tf.tensordot(
            layer_in, p_d_v[m], [[2], [0]]), n_heads)
        
        x_self_scores = tf.matmul(
            x_d_q, x_d_k, transpose_b=True)
        x_self_alphas = tf.nn.softmax(x_self_scores + attn_mask)
        
        x_self_conc = combine_heads(tf.matmul(x_self_alphas, x_d_v))
        x_self_head = tf.nn.dropout(tf.tensordot(
            x_self_conc, p_d_c[m], [[2], [0]]), rate=1.0-p_reg)
        
        if var_type == "norm_add":
            x_self_norm = tf.add(
                layer_in, layer_normalisation(
                    x_self_head, b_bias_1[m], b_scale_1[m]))
        elif var_type == "add_norm":
            x_self_norm = layer_normalisation(
                layer_in + x_self_head, b_bias_1[m], b_scale_1[m])
        
        # Encoder Decoder Attention Layer. #
        x_a_q = split_heads(tf.add(
            b_a_q[m, :step, :], tf.tensordot(
            x_self_norm, p_a_q[m], [[2], [0]])), n_heads)
        x_a_q = x_a_q * tf.math.rsqrt(head_size)
        
        x_a_k = split_heads(tf.tensordot(
            enc_outputs, p_a_k[m], [[2], [0]]) + b_a_k[m], n_heads)
        x_a_v = split_heads(tf.tensordot(
            enc_outputs, p_a_v[m], [[2], [0]]), n_heads)
        
        x_attn_scores = tf.matmul(
            x_a_q, x_a_k, transpose_b=True)
        x_attn_alphas = tf.nn.softmax(x_attn_scores)
        
        x_head_conc  = combine_heads(tf.matmul(x_attn_alphas, x_a_v))
        x_multi_head = tf.nn.dropout(tf.tensordot(
            x_head_conc, p_a_c[m], [[2], [0]]), rate=1.0-p_reg)
        
        if var_type == "norm_add":
            x_heads_norm = tf.add(
                x_self_norm, layer_normalisation(
                    x_multi_head, b_bias_2[m], b_scale_2[m]))
        elif var_type == "add_norm":
            x_heads_norm = layer_normalisation(
                x_self_norm + x_multi_head, b_bias_2[m], b_scale_2[m])
        
        # Feed forward layer. #
        x_ffw1 = tf.nn.relu(tf.tensordot(
            x_heads_norm, p_ff1[m], [[2], [0]]) + b_ff1[m])
        x_ffw2 = tf.tensordot(x_ffw1, p_ff2[m], [[2], [0]]) + b_ff2[m]
        x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
        
        # Residual Connection Layer. #
        if var_type == "norm_add":
            x_ffw_norm = tf.add(
                x_heads_norm, layer_normalisation(
                    x_ffwd, b_bias_3[m], b_scale_3[m]))
        elif var_type == "add_norm":
            x_ffw_norm = layer_normalisation(
                x_heads_norm + x_ffwd, b_bias_3[m], b_scale_3[m])
        
        # Append the output of the current layer. #
        layer_in = x_ffw_norm
    
    if var_type == "norm_add":
        dec_outputs = dec_inputs +\
            layer_normalisation(x_ffw_norm, o_bias, o_scale)
    elif var_type == "add_norm":
        dec_outputs = layer_normalisation(
            dec_inputs + x_ffw_norm, o_bias, o_scale)
    return dec_outputs

class TransformerNetwork(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, hidden_size, ffwd_size, 
        enc_vocab_size, dec_vocab_size, seq_enc, seq_dec, 
        embed_size=128, p_keep=0.9, var_type="norm_add", **kwargs):
        super(TransformerNetwork, self).__init__(**kwargs)
        
        self.p_keep = p_keep
        self.n_heads = n_heads
        self.var_type = var_type
        self.n_layers = n_layers
        self.seq_encode = seq_enc
        self.seq_decode = seq_dec
        self.hidden_size = hidden_size
        self.head_size   = int(hidden_size / n_heads)
        self.ffwd_size   = ffwd_size
        self.embed_size  = embed_size
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        
        self.W_enc_emb = tf.Variable(tf.random.normal(
            [self.enc_vocab_size, self.embed_size], stddev=0.1), name="e_emb")
        self.W_enc_lin = tf.Variable(tf.random.normal(
            [self.embed_size, self.hidden_size], stddev=0.1), name="e_lin")
        self.W_dec_emb = tf.Variable(tf.random.normal(
            [self.dec_vocab_size, self.embed_size], stddev=0.1), name="d_emb")
        self.W_dec_lin = tf.Variable(tf.random.normal(
            [self.embed_size, self.hidden_size], stddev=0.1), name="d_lin")
        
        self.p_enc_decode = tf.Variable(tf.random.normal(
            [self.hidden_size, self.seq_decode], stddev=0.1), name="p_enc_dec")
        self.p_out_decode = tf.Variable(tf.random.normal(
            [self.hidden_size, self.dec_vocab_size], stddev=0.1), name="d_out")
        
        # Transformer Encoder Variables. #
        e_pos_wgt_shape = [self.n_layers, self.seq_encode, self.hidden_size]
        d_pos_wgt_shape = [self.n_layers, self.seq_decode, self.hidden_size]
        
        attn_wgt_shape = [self.n_layers, self.hidden_size, self.hidden_size]
        ffwd_wgt_shape = [self.n_layers, self.hidden_size, self.ffwd_size]
        out_wgt_shape  = [self.n_layers, self.ffwd_size, self.hidden_size]
        
        self.e_pos_embed = tf.Variable(tf.random.normal(
            e_pos_wgt_shape, stddev=0.1), name="e_pos_embed")
        self.p_e_q = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_q")
        self.p_e_k = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_k")
        self.p_e_v = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_v")
        self.p_e_c = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_c")
        
        self.b_e_q = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_encode, self.hidden_size]), name="b_e_q")
        self.b_e_k = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_encode, self.hidden_size]), name="b_e_k")
        
        self.p_e_ff1 = tf.Variable(tf.random.normal(
            ffwd_wgt_shape, stddev=0.1), name="p_e_ff1") 
        self.p_e_ff2 = tf.Variable(tf.random.normal(
            out_wgt_shape, stddev=0.1), name="p_e_ff2")
        self.b_e_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_e_ff1")
        self.b_e_ff2 = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_ff2")
        
        self.e_i_bias  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="e_i_bias")
        self.e_i_scale = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="e_i_scale")
        self.e_o_bias  = tf.Variable(tf.zeros(
            [self.hidden_size]), name="e_o_bias")
        self.e_o_scale = tf.Variable(tf.ones(
            [self.hidden_size]), name="e_o_scale")
        
        self.b_e_bias_1  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_bias_1")
        self.b_e_bias_2  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_bias_2")
        self.b_e_scale_1 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_e_scale_1")
        self.b_e_scale_2 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_e_scale_2")
        
        # Transformer Decoder Variables. #
        self.d_pos_embed = tf.Variable(tf.random.normal(
            d_pos_wgt_shape, stddev=0.1), name="d_pos_embed")
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_c") 
        
        self.p_a_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_a_q")
        self.p_a_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_a_k")
        self.p_a_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_a_v")
        self.p_a_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_a_c")
        
        self.b_d_q  = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_decode, self.hidden_size]), name="b_d_q")
        self.b_d_k  = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_decode, self.hidden_size]), name="b_d_k")
        self.b_a_q  = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_decode, self.hidden_size]), name="b_a_q")
        self.b_a_k  = tf.Variable(tf.zeros(
            [self.n_layers, self.seq_encode, self.hidden_size]), name="b_a_k")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            ffwd_wgt_shape, stddev=0.1), name="p_d_ff1") 
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            out_wgt_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_d_ff2")
        
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
        self.b_d_bias_3  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_d_bias_3")
        self.b_d_scale_1 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_d_scale_1")
        self.b_d_scale_2 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_d_scale_2")
        self.b_d_scale_3 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_d_scale_3")
    
    def call(self, x_encode, x_decode):
        eps = 1.0e-6
        
        # Encoder. #
        x_enc_token = tf.nn.embedding_lookup(self.W_enc_emb, x_encode)
        x_dec_token = tf.nn.embedding_lookup(self.W_dec_emb, x_decode)
        x_enc_embed = tf.tensordot(x_enc_token, self.W_enc_lin, [[2], [0]])
        
        enc_dec_scores = tf.nn.relu(tf.transpose(tf.tensordot(
            x_enc_token, self.p_enc_decode, [[2], [0]]), [0, 2, 1])) + eps
        enc_dec_alphas = tf.divide(
            enc_dec_scores, tf.reduce_sum(
                enc_dec_scores, axis=[-1], keepdims=True))
        x_enc_decoder  = tf.matmul(enc_dec_alphas, x_enc_token)
        
        # Encoder Model. #
        enc_outputs = transformer_encode(
            self.n_layers, self.n_heads, 
            self.e_pos_embed, x_enc_embed, 
            self.b_e_q, self.b_e_k, 
            self.p_e_q, self.p_e_k, 
            self.p_e_v, self.p_e_c, 
            self.p_e_ff1, self.b_e_ff1, 
            self.p_e_ff2, self.b_e_ff2, 
            self.e_i_bias, self.e_i_scale, 
            self.b_e_bias_1, self.b_e_scale_1, 
            self.b_e_bias_2, self.b_e_scale_2, 
            self.e_o_bias, self.e_o_scale, 
            p_keep=self.p_keep, var_type=self.var_type)
        
        # Decoder Model. #
        x_dec_embed = tf.tensordot(
            x_dec_token + x_enc_decoder, self.W_dec_lin, [[2], [0]])
        dec_outputs = transformer_decode(
            self.seq_decode, self.n_layers, 
            self.n_heads, self.d_pos_embed, 
            x_dec_embed, enc_outputs, 
            self.b_d_q, self.b_d_k, 
            self.b_a_q, self.b_a_k, 
            self.p_d_q, self.p_d_k, 
            self.p_d_v, self.p_d_c, 
            self.p_a_q, self.p_a_k, 
            self.p_a_v, self.p_a_c, 
            self.p_d_ff1, self.b_d_ff1, 
            self.p_d_ff2, self.b_d_ff2, 
            self.d_i_bias, self.d_i_scale, 
            self.b_d_bias_1, self.b_d_scale_1, 
            self.b_d_bias_2, self.b_d_scale_2, 
            self.b_d_bias_3, self.b_d_scale_3, 
            self.d_o_bias, self.d_o_scale, 
            p_keep=self.p_keep, var_type=self.var_type)
        dec_logits  = tf.tensordot(
            dec_outputs, self.p_out_decode, [[2], [0]])
        return dec_logits
    
    def infer(self, x_encode, x_decode):
        # Encoder. #
        eps = 1.0e-6
        x_enc_token = tf.nn.embedding_lookup(self.W_enc_emb, x_encode)
        x_dec_token = tf.nn.embedding_lookup(self.W_dec_emb, x_decode)
        x_enc_embed = tf.tensordot(x_enc_token, self.W_enc_lin, [[2], [0]])
        
        enc_dec_scores = tf.nn.relu(tf.transpose(tf.tensordot(
            x_enc_token, self.p_enc_decode, [[2], [0]]), [0, 2, 1])) + eps
        enc_dec_alphas = tf.divide(
            enc_dec_scores, tf.reduce_sum(
                enc_dec_scores, axis=[-1], keepdims=True))
        
        x_enc_decoder = tf.matmul(enc_dec_alphas, x_enc_token)
        enc_outputs   = transformer_encode(
            self.n_layers, self.n_heads, 
            self.e_pos_embed, x_enc_embed, 
            self.b_e_q, self.b_e_k, 
            self.p_e_q, self.p_e_k, 
            self.p_e_v, self.p_e_c, 
            self.p_e_ff1, self.b_e_ff1, 
            self.p_e_ff2, self.b_e_ff2, 
            self.e_i_bias, self.e_i_scale, 
            self.b_e_bias_1, self.b_e_scale_1, 
            self.b_e_bias_2, self.b_e_scale_2, 
            self.e_o_bias, self.e_o_scale, 
            p_keep=1.0, var_type=self.var_type)
        
        # Inference. #
        infer_embed = [tf.expand_dims(x_dec_token[:, 0, :], axis=1)]
        infer_index = []
        for step in range(self.seq_decode):
            x_inf_inputs = tf.add(
                tf.concat(infer_embed, axis=1), 
                x_enc_decoder[:, :(step+1), :])
            x_inf_embed  = tf.tensordot(
                x_inf_inputs, self.W_dec_lin, [[2], [0]])
            
            tmp_outputs = transformer_decode(
                step+1, self.n_layers, 
                self.n_heads, self.d_pos_embed, 
                x_inf_embed, enc_outputs, 
                self.b_d_q, self.b_d_k, 
                self.b_a_q, self.b_a_k, 
                self.p_d_q, self.p_d_k, 
                self.p_d_v, self.p_d_c, 
                self.p_a_q, self.p_a_k, 
                self.p_a_v, self.p_a_c, 
                self.p_d_ff1, self.b_d_ff1, 
                self.p_d_ff2, self.b_d_ff2, 
                self.d_i_bias, self.d_i_scale, 
                self.b_d_bias_1, self.b_d_scale_1, 
                self.b_d_bias_2, self.b_d_scale_2, 
                self.b_d_bias_3, self.b_d_scale_3, 
                self.d_o_bias, self.d_o_scale, 
                p_keep=1.0, var_type=self.var_type)
            
            tmp_logit  = tf.matmul(
                tmp_outputs[:, -1, :], self.p_out_decode)
            tmp_argmax = tf.argmax(
                tmp_logit, axis=-1, output_type=tf.int32)
            next_embed = tf.matmul(
                tf.nn.softmax(tmp_logit), self.W_dec_emb)
            
            infer_index.append(tf.expand_dims(tmp_argmax, axis=1))
            infer_embed.append(tf.expand_dims(next_embed, axis=1))
        
        # Concatenate the list into a tensor. #
        infer_ids = tf.concat(infer_index, axis=1)
        return infer_ids


