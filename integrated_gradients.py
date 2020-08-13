#!/usr/bin/env python
# coding: utf-8

# ### Intergrated Gradients
# 
# From the paper [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365.pdf), Intergrated Gradients is a method for attribution of feature importance based on the line intergral of the output w.r.t the input dimensions over the straigthline path from the baseline to the input in question.
# 
# My implementation used the [Tensorflow tutorial](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients).

# In[4]:


import tensorflow as tf
from focal_loss import BinaryFocalLoss


# In[5]:


model = tf.keras.models.load_model('saved_models/CBoW_focal_loss_w_esports_9000')


# In[18]:


sum(sum(model.predict((tf.zeros((1,745))))))


# In[19]:


sum(sum(model(tf.zeros((1,745)))))


# In[10]:


def input_path_interpolation(baseline, input_vec, alphas):    
    alphas_x = alphas[:, tf.newaxis]
    input_x = tf.expand_dims(input_vec, axis=0)
    baseline_x = tf.expand_dims(baseline, axis=0)
    
    delta = input_x - baseline_x
    interpolated_vec = baseline_x + alphas_x * delta
    
    return interpolated_vec


# In[20]:


def compute_gradients(inputs, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        logits = model(inputs)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, inputs)


# In[26]:


def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


# In[28]:


@tf.function
def integrated_gradients(baseline,
                         input_vec,
                         target_class_idx,
                         m_steps=300,
                         batch_size=32):
    # 1. Generate alphas
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

    # Accumulate gradients across batches
    integrated_gradients = 0.0

    # Batch alpha vectors
    ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

    for batch in ds:

        # 2. Generate interpolated vectors
        batch_interpolated_inputs = input_path_interpolation(baseline=baseline,
                                                             input_vec=input_vec,
                                                             alphas=batch)

        # 3. Compute gradients between model outputs and interpolated inputs
        batch_gradients = compute_gradients(inputs=batch_interpolated_inputs,
                                            target_class_idx=target_class_idx)

        # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

    # 5. Scale integrated gradients with respect to input
    scaled_integrated_gradients = (input_vec - baseline) * integrated_gradients
    
    return scaled_integrated_gradients


# In[ ]:




