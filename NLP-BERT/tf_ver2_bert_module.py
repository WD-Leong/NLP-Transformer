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

class BERT_Network(tf.keras.Model):
    def __init__(
    self, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_dict, seq_length, 
    p_keep=0.9, p_reg=1.0, var_type="norm_add", **kwargs):
        super(BERT_Network, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type  = var_type
        self.seq_length = seq_length
        self.vocab_dict = vocab_dict
        self.vocab_size  = len(vocab_dict)
        self.hidden_size = hidden_size
        self.ffwd_size   = ffwd_size
        self.head_size   = int(hidden_size / n_heads)
        
        # Embedding matrices. #
        emb_enc_shape = [self.vocab_size, self.hidden_size]
        self.W_emb_enc = tf.Variable(tf.random.normal(
            emb_enc_shape, stddev=0.1), name="enc_embed")
        
        # BERT Variables. #
        attn_wgt_shape  = [
            self.n_layers, self.hidden_size, self.hidden_size]
        attn_ffw1_shape = [
            self.n_layers, self.hidden_size, self.ffwd_size]
        attn_ffw2_shape = [
            self.n_layers, self.ffwd_size, self.hidden_size]
        pos_embed_shape = [
            self.n_layers, self.seq_length, self.hidden_size]
        
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_c")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
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
    
    def transformer_encode(
        self, enc_inputs, p_reg=1.0, training=False):
        n_heads = self.n_heads
        head_size = tf.cast(self.head_size, tf.float32)
        
        if training:
            p_keep = self.p_keep
        else:
            p_keep = 1.0
        
        layer_input = enc_inputs
        for m in range(self.n_layers):
            layer_in = \
                self.x_emb_pos_dec[m, :, :] + layer_input
            
            # Self Attention Layer. #
            x_sq = split_heads(tf.tensordot(
                layer_in, self.p_d_q[m], [[2], [0]]), n_heads)
            x_sq = x_sq * tf.math.rsqrt(head_size)
    
            x_sk = split_heads(tf.tensordot(
                layer_in, self.p_d_k[m], [[2], [0]]), n_heads)
            x_sv = split_heads(tf.tensordot(
                layer_in, self.p_d_v[m], [[2], [0]]), n_heads)
            
            x_s_scores = tf.matmul(
                x_sq, x_sk, transpose_b=True)
            x_s_alphas = tf.nn.softmax(x_s_scores)
            
            x_self_conc  = combine_heads(tf.matmul(x_s_alphas, x_sv))
            x_multi_self = tf.nn.dropout(tf.tensordot(
                x_self_conc, self.p_d_c[m], [[2], [0]]), rate=1.0-p_reg)
            
            if self.var_type == "norm_add":
                x_self_norm = tf.add(layer_in, layer_normalisation(
                    x_multi_self, self.b_d_bias_1[m], self.b_d_scale_1[m]))
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(
                    tf.add(layer_in, x_multi_self), 
                    self.b_d_bias_1[m], self.b_d_scale_1[m])
    
            # Feed forward layer. #
            x_ffw1 = tf.nn.relu(tf.add(
                self.b_d_ff1[m], tf.tensordot(
                    x_self_norm, self.p_d_ff1[m], [[2], [0]])))
            x_ffw2 = tf.add(
                self.b_d_ff2[m], tf.tensordot(
                    x_ffw1, self.p_d_ff2[m], [[2], [0]]))
            x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
            
            if self.var_type == "norm_add":
                x_ffw_norm = tf.add(x_self_norm, layer_normalisation(
                    x_ffwd, self.b_d_bias_2[m], self.b_d_scale_2[m]))
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    tf.add(x_self_norm, x_ffwd), 
                    self.b_d_bias_2[m], self.b_d_scale_2[m])
            
            # Append the output. #
            layer_in = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            enc_outputs = tf.add(
                enc_inputs, layer_normalisation(
                    x_ffw_norm, self.d_o_bias, self.d_o_scale))
        elif self.var_type == "add_norm":
            enc_outputs = layer_normalisation(
                tf.add(enc_inputs, x_ffw_norm), 
                self.d_o_bias, self.d_o_scale)
        return enc_outputs
    
    def call(self, x_input, training=False):
        # Word or Sub-word embeddings. #
        x_enc_embed = tf.nn.embedding_lookup(
            self.W_emb_enc, x_input)
        
        # Training via Teacher forcing. #
        enc_outputs = self.transformer_encode(
            x_enc_embed, p_reg=1.0, training=training)
        
        enc_logits = tf.tensordot(
            enc_outputs, self.W_emb_enc, [[2], [1]])
        return enc_logits, enc_outputs

class BERT_Classifier(tf.keras.Model):
    def __init__(
    self, n_classes, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_dict, seq_length, 
    p_keep=0.9, p_reg=1.0, var_type="norm_add", **kwargs):
        super(BERT_Classifier, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.var_type  = var_type
        self.seq_length = seq_length
        self.vocab_dict = vocab_dict
        self.vocab_size  = len(vocab_dict)
        self.hidden_size = hidden_size
        self.ffwd_size   = ffwd_size
        self.head_size   = int(hidden_size / n_heads)
        
        # Embedding matrices. #
        emb_enc_shape = [self.vocab_size, self.hidden_size]
        self.W_emb_enc = tf.Variable(tf.random.normal(
            emb_enc_shape, stddev=0.1), name="enc_embed")
        
        # Output projection. #
        output_shape = [self.hidden_size, self.n_classes]
        self.p_output = tf.Variable(tf.random.normal(
            output_shape, stddev=0.1), name="p_output")
        
        # BERT Variables. #
        attn_wgt_shape  = [
            self.n_layers, self.hidden_size, self.hidden_size]
        attn_ffw1_shape = [
            self.n_layers, self.hidden_size, self.ffwd_size]
        attn_ffw2_shape = [
            self.n_layers, self.ffwd_size, self.hidden_size]
        pos_embed_shape = [
            self.n_layers, self.seq_length, self.hidden_size]
        
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_c")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
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
    
    def transformer_encode(
        self, enc_inputs, p_reg=1.0, training=False):
        n_heads = self.n_heads
        head_size = tf.cast(self.head_size, tf.float32)
        
        if training:
            p_keep = self.p_keep
        else:
            p_keep = 1.0
        
        layer_input = enc_inputs
        for m in range(self.n_layers):
            layer_in = \
                self.x_emb_pos_dec[m, :, :] + layer_input
            
            # Self Attention Layer. #
            x_sq = split_heads(tf.tensordot(
                layer_in, self.p_d_q[m], [[2], [0]]), n_heads)
            x_sq = x_sq * tf.math.rsqrt(head_size)
    
            x_sk = split_heads(tf.tensordot(
                layer_in, self.p_d_k[m], [[2], [0]]), n_heads)
            x_sv = split_heads(tf.tensordot(
                layer_in, self.p_d_v[m], [[2], [0]]), n_heads)
            
            x_s_scores = tf.matmul(
                x_sq, x_sk, transpose_b=True)
            x_s_alphas = tf.nn.softmax(x_s_scores)
            
            x_self_conc  = combine_heads(tf.matmul(x_s_alphas, x_sv))
            x_multi_self = tf.nn.dropout(tf.tensordot(
                x_self_conc, self.p_d_c[m], [[2], [0]]), rate=1.0-p_reg)
            
            if self.var_type == "norm_add":
                x_self_norm = tf.add(layer_in, layer_normalisation(
                    x_multi_self, self.b_d_bias_1[m], self.b_d_scale_1[m]))
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(
                    tf.add(layer_in, x_multi_self), 
                    self.b_d_bias_1[m], self.b_d_scale_1[m])
    
            # Feed forward layer. #
            x_ffw1 = tf.nn.relu(tf.add(
                self.b_d_ff1[m], tf.tensordot(
                    x_self_norm, self.p_d_ff1[m], [[2], [0]])))
            x_ffw2 = tf.add(
                self.b_d_ff2[m], tf.tensordot(
                    x_ffw1, self.p_d_ff2[m], [[2], [0]]))
            x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
            
            if self.var_type == "norm_add":
                x_ffw_norm = tf.add(x_self_norm, layer_normalisation(
                    x_ffwd, self.b_d_bias_2[m], self.b_d_scale_2[m]))
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    tf.add(x_self_norm, x_ffwd), 
                    self.b_d_bias_2[m], self.b_d_scale_2[m])
            
            # Append the output. #
            layer_in = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            enc_outputs = tf.add(
                enc_inputs, layer_normalisation(
                    x_ffw_norm, self.d_o_bias, self.d_o_scale))
        elif self.var_type == "add_norm":
            enc_outputs = layer_normalisation(
                tf.add(enc_inputs, x_ffw_norm), 
                self.d_o_bias, self.d_o_scale)
        return enc_outputs
    
    def call(self, x_input, training=False):
        # Word or Sub-word embeddings. #
        x_enc_embed = tf.nn.embedding_lookup(
            self.W_emb_enc, x_input)
        
        # Training via Teacher forcing. #
        enc_outputs = self.transformer_encode(
            x_enc_embed, p_reg=1.0, training=training)
        
        enc_logits = tf.matmul(
            enc_outputs[:, 0, :], self.p_output)
        return enc_logits, enc_outputs
    
    def infer(self, x_input):
        infer_logits = self.call(
            x_input, training=False)[0]
        infer_labels = tf.argmax(
            infer_logits, axis=1, output_type=tf.int32)
        return infer_labels

class BERT_FineTune(tf.keras.Model):
    def __init__(
    self, n_classes, bert_base_model, **kwargs):
        super(BERT_FineTune, self).__init__(**kwargs)
        self.n_classes  = n_classes
        self.bert_base  = bert_base_model
        self.p_classify = tf.keras.layers.Dense(
            n_classes, use_bias=False, name="p_classify")
    
    def call(self, x_input, training=False):
        enc_outputs = self.bert_base(
            x_input, training=training)[1]
        enc_labels  = self.p_classify(enc_outputs[:, 0, :])
        return enc_labels
    
    def infer(self, x_input):
        enc_outputs = self.bert_model(
            x_input, training=False)[1]
        
        pred_logits = self.p_classifier(enc_outputs[:, 0, :])
        label_pred  = tf.argmax(
            pred_logits, axis=-1, output_type=tf.int32)
        return label_pred
