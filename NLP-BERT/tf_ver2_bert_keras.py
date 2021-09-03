import tensorflow as tf

def scaled_dot_product_attention(
    q, k, v, mask=None, neg_infty=-1.0e9):
    # Head dimension. #
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    lq = tf.shape(q)[2]
    lk = tf.shape(k)[2]
    
    # Multiplicative Attention. #
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale multiplicative attention mechanism. #
    attn_logits = matmul_qk * tf.math.rsqrt(dk)
    
    # Add the mask to the attention mechanism. #
    if mask is not None:
        attn_mask = (mask * neg_infty)
    else:
        attn_mask = tf.zeros([lq, lk])
    attn_logits += attn_mask
    
    attn_weights = tf.nn.softmax(attn_logits, axis=-1)
    attn_outputs = tf.matmul(attn_weights, v)
    return attn_outputs, attn_weights

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def call(self, q, k, v, mask=None):
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        
        attn_tuple = scaled_dot_product_attention(
            q, k, v, mask=mask)
        
        attn_wgt = attn_tuple[1]
        attn_out = self.combine_heads(attn_tuple[0])
        attn_out = self.wc(attn_out)
        return attn_out, attn_wgt
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# Transformer Encoder Layer. #
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, rate=0.1):
        super(EncoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(d_model, n_heads)
        
        self.lnorm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6)
        self.lnorm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training=True):
        attn_self_tuple = self.attn_self(
            x, x, x, mask=None)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_tuple[0], training=training)
        attn_self_output = tf.add(
            x, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, 
        d_ffwd, vocab_size, max_seq_length, rate=0.1):
        super(Encoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate = rate
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        self.enc_embed = tf.keras.layers.Embedding(
            vocab_size, d_model)
        self.pos_embed = tf.keras.layers.Embedding(
            max_seq_length, d_model)
        
        # Encoder Layers. #
        tmp_enc_layers = []
        for n_layer in range(n_layers):
            tmp_enc_layers.append(EncoderLayer(
                d_model, n_heads, d_ffwd, rate))
        
        self.enc_layers = tmp_enc_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate)
        del tmp_enc_layers
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_pos_embed = self.pos_embed(x_pos_index)
        x_tok_embed = self.enc_embed(x) * self.d_rsqrt
        x_dec_embed = self.emb_dropout(tf.add(
            x_tok_embed, x_pos_embed), training=training)
        
        layer_input = x_dec_embed
        for m in range(self.n_layers):
            layer_output = self.enc_layers[m](
                layer_input, training=training)
            layer_input  = layer_output
        return layer_output

class BERTClassifier(tf.keras.Model):
    def __init__(
        self, n_classes, n_layers, n_heads, d_model, 
        d_ffwd, vocab_size, max_seq_length, rate=0.1):
        super(BERTClassifier, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate = rate
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        
        # BERT Network. #
        self.bert_model = Encoder(
            n_layers, d_model, n_heads, d_ffwd, 
            vocab_size, max_seq_length, rate=rate)
        self.p_decoder = tf.keras.layers.Dense(n_classes)
    
    def call(self, x_encode, training=True):
        enc_outputs = self.bert_model(
            x_encode, training=training)
        enc_logits  = self.p_decoder(
            enc_outputs[:, 0, :])
        return enc_logits, enc_outputs
    
    def infer(self, x):
        tmp_logit = self.call(x, training=False)[0]
        tmp_index = tf.argmax(
            tmp_logit, axis=1, output_type=tf.int32)
        return tmp_index
        

