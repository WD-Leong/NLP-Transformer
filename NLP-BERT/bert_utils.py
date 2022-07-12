import numpy as np
import tensorflow as tf

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, x_mask, 
    x_input, x_sequence, x_output, 
    optimizer, cls_flag=True, reg_emb=0.01, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_mask = x_mask[id_st:id_en, :]
        tmp_input  = x_input[id_st:id_en, :]
        tmp_label  = x_output[id_st:id_en]
        tmp_output = x_sequence[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model(
                tmp_input, training=True)
            
            class_logits = model_outputs[0]
            vocab_logits = model_outputs[1]
            bert_outputs = model_outputs[2]
            
            # Masked Language Model Loss. #
            msk_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=vocab_logits), tmp_mask)
            num_mask = tf.cast(
                tf.reduce_sum(tmp_mask, axis=1), tf.float32)
            msk_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_xent, axis=1), num_mask))
            
            # CLS token is the average embeddings since there is #
            # no Next Sentence Prediction in this pre-training.  #
            cls_embed  = bert_outputs[:, 0, :]
            avg_embed  = tf.reduce_mean(
                bert_outputs[:, 1:, :], axis=1)
            emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(cls_embed - avg_embed), axis=1))
            
            # Supervised Loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_label, logits=class_logits))
            
            # Full Loss Function. #
            if cls_flag:
                tmp_losses = tf.add(
                    reg_emb*emb_losses, tf.add(
                        cls_losses, 0.0 * msk_losses))
            else:
                tmp_losses = tf.add(
                    reg_emb*emb_losses, tf.add(
                        0.0 * cls_losses, msk_losses))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses

def generate_input(
    tmp_text, word_2_idx, p_mask, 
    seq_length, replace_vocab, CLS_token, 
    EOS_token, MSK_token, TRU_token, UNK_token):
    tmp_i_tok = [word_2_idx.get(
        x, UNK_token) for x in tmp_text]
    num_token = len(tmp_i_tok)

    # Truncate the sequence if it exceeds the maximum #
    # sequence length. Randomly select the review's   #
    # start and end index to be the positive example. #
    if num_token > seq_length:
        # For the anchor. #
        id_st = np.random.randint(
            0, num_token-seq_length)
        id_en = id_st + seq_length
        
        tmp_i_idx = [CLS_token]
        tmp_i_idx += tmp_i_tok[id_st:id_en]
        
        if id_en < num_token:
            # Add TRUNCATE token. #
            tmp_i_idx += [TRU_token]
        else:
            tmp_i_idx += [EOS_token]
        del id_st, id_en
    else:
        tmp_i_idx = [CLS_token]
        tmp_i_idx += tmp_i_tok
        tmp_i_idx += [EOS_token]
    n_input = len(tmp_i_idx)

    # Generate the masked sequence. #
    mask_seq  = [MSK_token] * n_input
    tmp_mask  = np.random.binomial(
        1, p_mask, size=n_input)
    
    tmp_noise = [CLS_token]
    tmp_noise += list(np.random.choice(
        replace_vocab, size=n_input-2))
    tmp_noise += [tmp_i_idx[-1]]
    
    tmp_unif = np.random.uniform()
    if tmp_unif <= 0.8:
        # Replace with MASK token. #
        tmp_i_msk = [
            tmp_i_idx[x] if tmp_mask[x] == 0 else \
                mask_seq[x] for x in range(n_input)]
    elif tmp_unif <= 0.9:
        # Replace with random word. #
        tmp_i_msk = [
            tmp_i_idx[x] if tmp_mask[x] == 0 else \
                tmp_noise[x] for x in range(n_input)]
    else:
        # No replacement. #
        tmp_i_msk = tmp_i_idx
    return tmp_i_msk, tmp_i_idx, tmp_mask

