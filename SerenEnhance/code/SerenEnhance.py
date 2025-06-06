# load necessary packages
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import Model
from sklearn.metrics.pairwise import pairwise_kernels



# define data loader
class DataBuilder(tf.keras.utils.Sequence):
    def __init__(self, file_dir, t_path, r_path, u_path, batch_size, n_reviews, user_threshold, test=False):
        # other parameters
        self.file_dir = file_dir
        self.t_path = t_path
        self.r_path = r_path
        self.u_path = u_path
        self.user_list = np.array(os.listdir(file_dir))
        self.user_list_length = len(self.user_list)
        self.indexes = np.arange(self.user_list_length)
        self.batch_size = batch_size
        self.n_reviews = n_reviews
        self.user_threshold = user_threshold
        self.test = test

    def __len__(self):
        # Denotes the number of batches per epoch
        if self.test:
          return int((len(self.user_list) * (1-self.user_threshold)) // self.batch_size + 1)
        else:
          return int((len(self.user_list) * self.user_threshold) // self.batch_size + 1)


    def __getitem__(self, index):
        # Generate indexes of the batch
        last_index = (index + 1) * self.batch_size
        if self.test:
          if last_index < self.user_list_length*(1-self.user_threshold):
              indexes = self.indexes[(-1 * last_index)-1: (-1 * index * self.batch_size)-1]
          else:
              indexes = self.indexes[(-1 * int(self.user_list_length * (1-self.user_threshold)))-1 : (-1 * index * self.batch_size)-1]
        else:
          if last_index < (self.user_list_length*self.user_threshold):
              indexes = self.indexes[index * self.batch_size:last_index]
          else:
              indexes = self.indexes[index * self.batch_size:]
        return self.__dataGeneration(self.user_list[indexes])


    def __dataGeneration(self, user_list_batch):
        # read user csv
        users_arr = tf.zeros((0, self.n_reviews, 128), dtype=tf.float32)
        ts_arr = tf.zeros((0, 100, 128), dtype=tf.float32)
        rs_arr = tf.zeros((0, 100, 128), dtype=tf.float32)
        us_arr = tf.zeros((0, 100, 128), dtype=tf.float32)
        for user in user_list_batch:
            user_arr = np.genfromtxt(os.path.join(self.file_dir, user), delimiter=',', dtype=np.float32)  # array(r, 128)
            user_arr = np.expand_dims(self.sliceOrPad(user_arr, self.n_reviews), axis=0)  # array(1, n_reviews, 128)
            users_arr = np.concatenate([users_arr, user_arr], axis=0)
            t_arr = np.genfromtxt(os.path.join(self.t_path, user), delimiter=',', dtype=np.float32)#[100,128 ]
            t_arr = np.expand_dims(self.sliceOrPad(t_arr[1:], 100), axis=0)#[1,100,128 ]

            t_pos = tf.zeros((0, 128), dtype=tf.float32)
            t_neg = tf.zeros((0, 128), dtype=tf.float32)
            for i in range(len(t_arr)):
              for j in range(len(t_arr[i])):
                if j%2!=0:
                  t_neg = np.concatenate([t_neg, t_arr[i][j].reshape((1,-1))], axis=0)#[50,128 ]
                else:
                  t_pos = np.concatenate([t_pos, t_arr[i][j].reshape((1,-1))], axis=0)#[50,128 ]
              t_arr_pn = np.concatenate([t_pos, t_neg], axis=1)#[50,128*2 ]
              ts_arr = np.concatenate([ts_arr, t_arr_pn.reshape((1,-1,128))], axis=0)#[1,100=50pos+50neg,128 ]

            r_arr = np.genfromtxt(os.path.join(self.r_path, user), delimiter=',', dtype=np.float32)
            r_arr = np.expand_dims(self.sliceOrPad(r_arr[1:], 100), axis=0)
            r_pos = tf.zeros((0, 128), dtype=tf.float32)
            r_neg = tf.zeros((0, 128), dtype=tf.float32)
            for i in range(len(r_arr)):
              for j in range(len(r_arr[i])):
                if j%2!=0:
                  r_neg = np.concatenate([r_neg, r_arr[i][j].reshape((1,-1))], axis=0)
                else:
                  r_pos = np.concatenate([r_pos, r_arr[i][j].reshape((1,-1))], axis=0)
              r_arr_pn = np.concatenate([r_pos, r_neg], axis=1)
              rs_arr = np.concatenate([rs_arr, r_arr_pn.reshape((1,-1,128))], axis=0)

            u_arr = np.genfromtxt(os.path.join(self.u_path, user), delimiter=',', dtype=np.float32)[:,1:129]
            u_arr = np.expand_dims(self.sliceOrPad(u_arr[1:], 100), axis=0)
            u_pos = tf.zeros((0, 128), dtype=tf.float32)
            u_neg = tf.zeros((0, 128), dtype=tf.float32)
            for i in range(len(u_arr)):
              for j in range(len(u_arr[i])):
                if j>49:
                  u_neg = np.concatenate([u_neg, u_arr[i][j].reshape((1,-1))], axis=0)# (50, 128)
                else:
                  u_pos = np.concatenate([u_pos, u_arr[i][j].reshape((1,-1))], axis=0)# (50, 128)
              u_arr_pn = np.concatenate([u_pos, u_neg], axis=1)# (50, 256)
              us_arr = np.concatenate([us_arr, u_arr_pn.reshape((1,-1,128))], axis=0)

        return users_arr, ts_arr, rs_arr, us_arr
        # users_arr(batch_size=usernum, n_review=20, dim=128)
        # ts_arr[batch_size=usernum,100=50pos+50neg,128]
        # us_arr(batch_size=usernum,100=50pos+50neg, dim=128)

    @staticmethod
    def sliceOrPad(arr, threshold):
        row_num = arr.shape[0]
        if row_num < threshold:
            return np.pad(arr, ((threshold - row_num, 0), (0, 0)), 'constant')
        elif row_num > threshold:
            return arr[(-threshold-1):-1, :]
        else:
            return arr

# prepare data path
user_data_path = '../data/user_pre_encoded_sorted' # item embeddings
rel_path = '../data/user_rel_train'  # relevance samples
unp_path = '../data/user_unexpectedness_samples'  # unexpectedness samples
sre_path = '../data/user_pre_candidate' # serendipity samples
test_path = '../data/user_pre_test'  # test set

user_batch_size = 32
user_n_reviews = 20  # Only consider the last n reviews of a user
embed_dim = 128  # Embedding size for each token
n_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
user_threshold = 0.8 # split dataset

# from databuilder import DataBuilder
# loading training data
data_all = DataBuilder(user_data_path, sre_path, rel_path, unp_path, user_batch_size, user_n_reviews, user_threshold)
data_train = data_all[0][0]# users_arr(batch_size=usernum, n_review=20, dim=128)
sre_train = data_all[0][1]# ts_arr[batch_size=usernum,100=50pos+50neg,128]
rel_train = data_all[0][2]
unp_train = data_all[0][3]# us_arr(batch_size=usernum,100=50pos+50neg, dim=128)
for i in range(1, len(data_all)-1):
  data_train = np.concatenate([data_train, data_all[i][0]], axis=0)
  sre_train = np.concatenate([sre_train, data_all[i][1]], axis=0)
  rel_train = np.concatenate([rel_train, data_all[i][2]], axis=0)
  unp_train = np.concatenate([unp_train, data_all[i][3]], axis=0)

# loading test data
data_all = DataBuilder(user_data_path, test_path, rel_path, unp_path, user_batch_size, user_n_reviews, user_threshold, test=True)
data_test = data_all[0][0]
sre_test = data_all[0][1]
rel_test = data_all[0][2]
unp_test = data_all[0][3]
for i in range(1, len(data_all)):
  data_test = np.concatenate([data_test, data_all[i][0]], axis=0)
  sre_test  = np.concatenate([sre_test, data_all[i][1]], axis=0)
  rel_test  = np.concatenate([rel_test, data_all[i][2]], axis=0)
  unp_test  = np.concatenate([unp_test, data_all[i][3]], axis=0)

# transformer block
class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, n_heads, ff_dim, rate=0.1):
        super().__init__()
        # Multi-Head Attention
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim)
        # Feed Forward
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        # Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, training=True):
        # Part 1. Multi-Head Attention + Normalization + Residual
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # Part 2. Feed Forward + Normalization + Residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# position encoding
class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, n_reviews, embed_dim):
        super().__init__()
        # Compute the positional encodings
        self.pe = tf.Variable(tf.zeros((n_reviews, embed_dim), dtype=tf.float32), dtype=tf.float32)
        position = tf.expand_dims(tf.range(0, n_reviews, 1, dtype=tf.float32), axis=1)
        div_term = tf.math.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / embed_dim))
        term = tf.math.multiply(position, div_term)
        self.pe = self.pe[:, 0::2].assign(tf.math.sin(term))
        self.pe = self.pe[:, 1::2].assign(tf.math.cos(term))
        self.pe = tf.expand_dims(self.pe, axis=0)

    def __call__(self, x):
        return x + tf.broadcast_to(self.pe, tf.shape(x))

# Main function
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    embed_dim = 128  # Embedding size for position
    n_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    self.num_blocks = 3 # Number of stacking Trm blocks
    n_reviews = 20
    dim = 128 # dimension for clusters/items/preferences

    self.d1 = layers.Dense(dim, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))
    self.d2 = layers.Dense(dim, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))
    self.d3 = layers.Dense(dim, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))

    self.embedding_layer = TokenAndPositionEmbedding(n_reviews, embed_dim)
    self.transformer_block1 = TransformerBlock(embed_dim, n_heads, ff_dim)
    self.transformer_block2 = TransformerBlock(embed_dim, n_heads, ff_dim)
    self.transformer_block3 = TransformerBlock(embed_dim, n_heads, ff_dim)

    self.a = 0.6
    self.dis = 0.6


  def call(self, x, y_seren, y_rel, y_unp):
      #(data_test, sre_test, rel_test, unp_test, training=True)
    # position embedding
    x = self.embedding_layer(x)
    # serendipity
    for i in range(self.num_blocks):
      x_seren = self.transformer_block(x)
    x_seren = layers.GlobalAveragePooling1D()(x_seren)
    x_seren = self.d1(x_seren)
    x_seren = layers.Dropout(0.2)(x_seren)
    x_seren = tf.reshape(x_seren,[-1,1,x_seren.shape[-1]])

    r_seren = tf.matmul(x_seren, y_seren, transpose_b=True)
    r_seren = tf.reshape(r_seren,[-1,r_seren.shape[-1]])
    r_seren = tf.keras.activations.sigmoid(r_seren)
    r_seren_pos = r_seren[:,:50]
    r_seren_neg = r_seren[:,50:]
    # pair-wise
    pair_seren_r = r_seren_pos - r_seren_neg

    # Relevance
    for i in range(self.num_blocks):
      x_rel = self.transformer_block2(x)
    x_rel = layers.GlobalAveragePooling1D()(x_rel)
    x_rel = self.d2(x_rel)
    x_rel = layers.Dropout(0.2)(x_rel)
    x_rel = tf.reshape(x_rel,[-1,1,x_rel.shape[-1]])
    r_rel = tf.matmul(x_rel, y_rel, transpose_b=True)
    r_rel = tf.reshape(r_rel,[-1,r_rel.shape[-1]])
    r_rel = tf.keras.activations.sigmoid(r_rel)
    r_rel_pos = r_rel[:,:50]
    r_rel_neg = r_rel[:,50:]
    # pair-wise
    pair_rel_r = r_rel_pos - r_rel_neg

    # Unexpectedness
    for i in range(self.num_blocks):
      x_unp = self.transformer_block3(x)
    x_unp = layers.GlobalAveragePooling1D()(x_unp)
    x_unp = self.d2(x_unp)
    x_unp = layers.Dropout(0.2)(x_unp)
    x_unp = tf.reshape(x_unp,[-1,1,x_unp.shape[-1]])

    r_unp = tf.matmul(x_unp, y_unp, transpose_b=True)
    r_unp = tf.reshape(r_unp,[-1,r_unp.shape[-1]])
    r_unp = tf.keras.activations.sigmoid(r_unp)
    r_unp_pos = r_unp[:,:50]
    r_unp_neg = r_unp[:,50:]
    # pair-wise
    pair_unp_r = r_unp_pos - r_unp_neg

    # merging layer
    r = x_seren + self.a * x_rel + (1-self.a) * x_unp
    r = tf.matmul(r, y_seren, transpose_b=True)
    r = tf.nn.softmax(tf.reshape(r,[-1,r.shape[-1]]))

    return r, pair_seren_r, pair_rel_r, pair_unp_r

# Create an instance of the model
model = MyModel()

# optimizer
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='test_accuracy1')
test_accuracy_5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='test_accuracy5')
test_accuracy_10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='test_accuracy10')

# serendipity prepare
test_labels = np.zeros((sre_test.shape[0],sre_test.shape[1]))#[batch_size=usernum,1+99]
test_labels[:,0]=1
test_labels = np.reshape(test_labels,(sre_test.shape[0],sre_test.shape[1],1))#[batch_size=usernum,1+99,1]
test_y = np.concatenate((sre_test,test_labels),axis=2)#[batch_size=usernum,100=1pos+99neg,128+1]
test_y = np.array(test_y, dtype=np.float32)

# relevance prepare
rel_test_labels = np.zeros((rel_test.shape[0],rel_test.shape[1]))
rel_test_labels[:,0]=1
rel_test_labels = np.reshape(rel_test_labels,(rel_test.shape[0],rel_test.shape[1],1))
rel_test_y = np.concatenate((rel_test,rel_test_labels),axis=2)
rel_test_y = np.array(rel_test_y, dtype=np.float32)

# unexpectedness prepare
unp_test_labels = np.zeros((unp_test.shape[0],unp_test.shape[1]))
unp_test_labels[:,0]=1
unp_test_labels = np.reshape(unp_test_labels,(unp_test.shape[0],unp_test.shape[1],1))
unp_test_y = np.concatenate((unp_test,unp_test_labels),axis=2)
unp_test_y = np.array(unp_test_y, dtype=np.float32)

# training labels
# serendipity
train_labels = np.zeros((sre_train.shape[0],sre_train.shape[1]))
train_labels[:,:50] = 1
# relevance
rel_train_labels = np.zeros((rel_train.shape[0],rel_train.shape[1]))
rel_train_labels[:,:50] = 1
# unexpectedness
unp_train_labels = np.zeros((unp_train.shape[0],unp_train.shape[1]))
unp_train_labels[:,:50] = 1

train_ds = tf.data.Dataset.from_tensor_slices((data_train, sre_train, train_labels, rel_train, rel_train_labels, unp_train, unp_train_labels)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((data_test, test_y[:,:,:-1], test_y[:,:,-1], rel_test_y[:,:,:-1], rel_test_y[:,:,-1], unp_test_y[:,:,:-1], unp_test_y[:,:,-1])).batch(32)

# training step
def train_step(data_train, sre_train, sre_labels, rel_train, rel_labels, unp_train, unp_labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions, pair_sre, pair_rel, pair_unp = model(data_train, sre_train, rel_train, unp_train, training=True)

    loss_sre = loss_object(sre_labels[:,:50], pair_sre)
    loss_rel = loss_object(rel_labels[:,:50], pair_rel)
    loss_unp = loss_object(unp_labels[:,:50], pair_unp)

    # joint loss
    loss = loss_sre + loss_rel + loss_unp
    # print(loss)

  gradients = tape.gradient(loss, model.trainable_variables)
  # print(gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

# test step
from sklearn.metrics import ndcg_score

def test_step(data_test, sre_test, sre_labels, rel_test, rel_labels, unp_test, unp_labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions, pair_sre, pair_rel, pair_unp = model(data_test, sre_test, rel_test, unp_test, training=True)
  t_loss = loss_object(sre_labels, predictions)

  test_loss(t_loss)
  test_accuracy_1(sre_labels, predictions)
  test_accuracy_5(sre_labels, predictions)
  test_accuracy_10(sre_labels, predictions)

  return sre_labels.tolist(), predictions.numpy().tolist()

EPOCHS = 100

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  test_loss.reset_states()
  test_accuracy_1.reset_states()
  test_accuracy_5.reset_states()
  test_accuracy_10.reset_states()

  seren_truth = []
  seren_pred = []

  for data, sre_train, sre_label, rel_train, rel_label, unp_train, unp_label in train_ds:
    train_step(data, sre_train, sre_label, rel_train, rel_label, unp_train, unp_label)
#test_ds = tf.data.Dataset.from_tensor_slices((data_test, test_y[:,:,:-1], test_y[:,:,-1],
  # rel_test_y[:,:,:-1], rel_test_y[:,:,-1], unp_test_y[:,:,:-1], unp_test_y[:,:,-1])).batch(32)
  for test_data, sre_test, sre_label, rel_test, rel_label, unp_test, unp_label in test_ds:
    test_truth, test_pred = test_step(test_data, sre_test, sre_label, rel_test, rel_label, unp_test, unp_label)
    seren_truth += test_truth
    seren_pred += test_pred

  ndcg_5 = ndcg_score(seren_truth, seren_pred, k=5)
  ndcg_10 = ndcg_score(seren_truth, seren_pred, k=10)


  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test HR@1: {test_accuracy_1.result() * 100},'
    f'Test HR@5: {test_accuracy_5.result() * 100},'
    f'Test HR@10: {test_accuracy_10.result() * 100},'
    'Test NDCG@5: ', ndcg_5,
    'Test NDCG@10: ', ndcg_10,
  )
