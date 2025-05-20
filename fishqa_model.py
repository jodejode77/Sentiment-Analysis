import numpy as np

def length(sequences):
    used = tf.cast(tf.math.not_equal(sequences, 0), tf.float32)
    seq_len = tf.reduce_sum(used, axis=1)
    return tf.cast(seq_len, tf.int32)

def read_question():
    with open('q1_data', 'rb') as f:
        q1 = pickle.load(f)
    with open('q2_data', 'rb') as g:
        q2 = pickle.load(g)
    with open('q3_data', 'rb') as b:
        q3 = pickle.load(b)
    return [q1, q2, q3]

def shuffle_data(x, y):
    indices = np.random.permutation(len(x))
    return [x[i] for i in indices], [y[i] for i in indices]

class FISHQA(tf.keras.Model):
    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=64, dropout_keep_proba=0.5, query=[]):
        super(FISHQA, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_keep_proba = dropout_keep_proba
        self.query = query

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.query_projector = tf.keras.layers.Dense(hidden_size * 2, activation='tanh')

        self.gru_fw = tf.keras.layers.GRU(hidden_size, return_sequences=True)
        self.gru_bw = tf.keras.layers.GRU(hidden_size, return_sequences=True, go_backwards=True)
        self.dropout = tf.keras.layers.Dropout(rate=1 - dropout_keep_proba)
        self.classifier = tf.keras.layers.Dense(num_classes)
        self.context_vector = self.add_weight(
            name="context_vector",
            shape=(hidden_size * 2,),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            trainable=True
        )

        self.att_dense_1 = tf.keras.layers.Dense(hidden_size * 2, activation='tanh')
        self.att_dense_2 = tf.keras.layers.Dense(hidden_size * 2, activation='tanh')
        self.att_dense_3 = tf.keras.layers.Dense(hidden_size * 2, activation='tanh')
        self.att_dense_4 = tf.keras.layers.Dense(hidden_size * 2, activation='tanh')

    def build(self, input_shape):
        B, S, F = input_shape
        dummy_input = tf.zeros((B, S, F))
        self.call(dummy_input, training=False)
        super().build(input_shape)

    def call(self, input_x, training=False):
        B = tf.shape(input_x)[0]
        S = tf.shape(input_x)[1]
        W = tf.shape(input_x)[2]

        x = tf.reshape(input_x, [-1, W])
        embedded = self.embedding(x)

        q1_emb = self._query_embedding(self.query[0])
        q2_emb = self._query_embedding(self.query[1])
        q3_emb = self._query_embedding(self.query[2])

        word_encoded = self._bi_gru(embedded)
        sent_vec, _ = self._attention(word_encoded, q1_emb, q2_emb, q3_emb)

        sent_vec = self.dropout(sent_vec, training=training)
        sent_vec = tf.reshape(sent_vec, [B, S, self.hidden_size * 2])

        doc_encoded = self._bi_gru(sent_vec)
        doc_vec, _ = self._attention(doc_encoded, q1_emb, q2_emb, q3_emb)

        doc_vec = self.dropout(doc_vec, training=training)
        logits = self.classifier(doc_vec)
        return logits

    def _query_embedding(self, query):
        emb = self.embedding(tf.constant(query, dtype=tf.int32))  # [seq_len, emb_dim]
        avg = tf.reduce_mean(emb, axis=0)  # [emb_dim]
        avg = tf.expand_dims(avg, axis=0)  # [1, emb_dim] to make it 2D for Dense
        projected = self.query_projector(avg)  # [1, hidden_size * 2]
        return tf.squeeze(projected, axis=0)   # back to [hidden_size * 2]

    
    def _bi_gru(self, inputs):
        fw = self.gru_fw(inputs)
        bw = self.gru_bw(inputs)
        return tf.concat([fw, bw], axis=-1)

    def _attention(self, inputs, q1, q2, q3):
        context = self.context_vector

        h1 = self.att_dense_1(inputs)
        h2 = self.att_dense_2(inputs)
        h3 = self.att_dense_3(inputs)
        h4 = self.att_dense_4(inputs)

        def compute_alpha(h, query):
            query = tf.reshape(query, [1, 1, -1])
            query = tf.broadcast_to(query, tf.shape(h))
            score = tf.reduce_sum(h * query, axis=-1, keepdims=True)
            return tf.nn.softmax(score, axis=1)

        t_alpha = compute_alpha(h1, context)
        q_alpha1 = compute_alpha(h2, q1)
        q_alpha2 = compute_alpha(h3, q2)
        q_alpha3 = compute_alpha(h4, q3)

        alpha = (t_alpha + q_alpha1 + q_alpha2 + q_alpha3) / 4
        output = tf.reduce_sum(inputs * alpha, axis=1)
        return output, alpha