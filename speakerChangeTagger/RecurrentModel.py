'''
A recurrent neural network for speaker change detection
not implemented yet
'''
import tensorflow as tf

class RecurrentModel:
    def __init__(self, args, textData):
        self.args = args
        self.textData = textData

        self.input_utterances = None
        self.labels = None
        self.dropOutRate = None
        self.embedded = None
        self.length = None
        self.predictions = None

        self.loss = None

        self.correct = None
        self.accuracy = None
        self.correct_predictions = None

        # this is for debug
        self.vectors = None
        self.batchSize = None
        self.optOp = None

        self.buildNetWork()

    def buildNetWork(self):
        # last_outputs is of shape [fake_batch_size, self.args.wordUnits]
        with tf.variable_scope('word'):
            last_outputs = self._buildWordNetwork()
        with tf.variable_scope('utterance'):
            context_vectors = self._buildUttNetwork(last_outputs)
        # use context_vectors_squeezed as inputs to softmax
        # shape: [true_batch_size, self.args.uttUnits]

        with tf.name_scope('output'):
            context_vectors_squeezed = tf.squeeze(context_vectors, name='context_squeezed', axis=[1])
            # note: slice context_vector_squeezed use self.batchSize
            context_vectors_squeezed_slice = tf.slice(context_vectors_squeezed, begin=[0, 0],
                                                      size=[self.batchSize, self.args.uttUnits], name='true_sf')

            weights = tf.Variable(tf.truncated_normal([self.args.uttUnits, self.args.numClasses], stddev=0.5),
                                  name='weights')
            biases = tf.Variable(tf.truncated_normal([self.args.numClasses], stddev=0.5), name='biases')
            logits = tf.add(tf.matmul(context_vectors_squeezed_slice, weights), biases, name='logits')

            self.predictions = tf.argmax(logits, axis=1, name='predictions')
            self.vectors = self.predictions
        with tf.name_scope('loss'):
            # note: since we have the sparse version, we don't need to have one-hot labels
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels, name='loss_')
            self.loss = tf.reduce_mean(loss_, name='loss')
        with tf.name_scope('evaluation'):
            self.correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
            self.correct = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32), name='numCorrect')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')

        with tf.name_scope('backpropagation'):
            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.minimize(self.loss)



    def _buildWordNetwork(self):
        # in this implementation, we still have fake batch size in the input_utterances
        # fake batch_size = (self.args.uttContextSize - 1) + batch_size
        # for simplicity, use self.args.uttWindowSize for self.args.uttContextSize
        self.input_utterances = tf.placeholder(tf.int32, [None, self.args.maxLength], name='input_utterances')
        self.length = tf.placeholder(tf.int32, [None], name='sequence_length')
        # self.labels use true batch_size
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')
        self.batchSize = tf.placeholder(tf.int32, (), name='true_batch_size')


        with tf.name_scope('embeddingLayer'):
            # whether or not to use the pretrained embeddings
            if self.args.preEmbedding == False:
                embeddings = tf.Variable(
                    tf.truncated_normal([self.textData.getVocabularySize()-1, self.args.embeddingSize], stddev=0.5),
                    name='embeddings')
            else:
                embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding')
            # note: for <pad>, embedding should be all zeros
            zero_embedding = tf.Variable(tf.zeros([1, self.args.embeddingSize]), name='padEmbedding', trainable=False)
            embeddings = tf.concat(concat_dim=0, values=[zero_embedding, embeddings])
            self._variableSummaries(embeddings)
            # self.embedded is a 3-dimentional matrix of shape [fake_batch_size, maxLength, embeddingSize]
            self.embedded = tf.nn.embedding_lookup(embeddings, self.input_utterances)

        with tf.name_scope('sentence_encoder'):
            with tf.name_scope('cell'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.args.wordUnits, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropOutRate,
                                                         output_keep_prob=self.dropOutRate)
                multiCell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.args.wordLayers, state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell=multiCell, inputs=self.embedded,
                                                sequence_length=self.length, dtype=tf.float32)

            # note: use tf.gather_nd to replace this function when gradient is implemented
            def last_relevant(output, length):
                batch_size = tf.shape(output)[0]
                max_length = tf.shape(output)[1]
                out_size = int(output.get_shape()[2])
                index = tf.range(0, batch_size) * max_length + (length - 1)
                flat = tf.reshape(output, [-1, out_size])
                relevant = tf.gather(flat, index, name='last_outputs')
                return relevant

            # last_outputs is of shape [fake_batch_size, self.args.wordUnits]
            last_outputs = last_relevant(outputs, self.length)
            return last_outputs

    def _buildUttNetwork(self, last_outputs):
        '''
        :param last_outputs: of shape [fake_batch_size, self.args.wordUnits]
        :return: softmax input for every sentence, [batch_size, self.args.uttUnits]
        '''
        # note: do not access self.args.batchSize directly

        utt_inputs = []
        for i in range(self.args.batchSize):
            utt_input = tf.slice(last_outputs, begin=[i, 0], size=[self.args.uttWindowSize, self.args.wordUnits])
            utt_inputs.append(utt_input)

            # cheers! Now utt_inputs_pack is of shape [true_batch_size, self.args.uttWindowSize, self.args.wordUnits]
            # sen_0, sent_1, sent_2, ......, sent_cur
            # |-------context(windowSize--)----------|
        #note: for batch at the end of a show, some in the utt_inputs may be useless, slice it before feed into softmax
        utt_inputs_pack = tf.pack(utt_inputs)

        with tf.name_scope('context_encoder'):
            with tf.name_scope('cell'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.args.uttUnits, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropOutRate,
                                                         output_keep_prob=self.dropOutRate)
                multiCell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.args.uttLayers, state_is_tuple=True)
            # outputs is of shape [true_batch_size, self.args.uttWindowSize, self.args.uttUnits]
            outputs, states = tf.nn.dynamic_rnn(cell=multiCell, inputs=utt_inputs_pack, dtype=tf.float32)

            # last_outputs is of shape [true_batch_size, 1, self.args.uttUnits]
            last_outputs = tf.slice(outputs, begin=[0, self.args.uttWindowSize-1, 0], size=[self.batchSize, 1, self.args.uttUnits])

            return last_outputs

    def step(self, batch, test = False):
        feed_dict = {}
        ops = None
        # for batch at the end of a show, its length may be shorter than other batches, in RNN implementation, pad them
        def pad(sequence, length):
            padding = []
            for i in range(self.args.maxLength):
                padding.append(0)
            while len(sequence) < self.args.batchSize + 2*(self.args.uttWindowSize-1) :
                sequence.append(padding)
            while len(length) < self.args.batchSize + 2 * (self.args.uttWindowSize - 1):
                length.append(2)
            return sequence, length

        if len(batch.sequence) < self.args.batchSize + 2*(self.args.uttWindowSize-1):
            batch.sequence, batch.length = pad(batch.sequence, batch.length)
        if len(batch.length) < self.args.batchSize + 2*(self.args.uttWindowSize-1):
            batch.sequence, batch.length = pad(batch.sequence, batch.length)

        assert len(batch.sequence) == len(batch.length)

        feed_dict[self.input_utterances] = batch.sequence
        feed_dict[self.labels] = batch.labels
        feed_dict[self.batchSize] = len(batch.labels)
        feed_dict[self.length] = batch.length
        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.correct, self.predictions, self.vectors)
        else:
            # during test, do not use drop out!!!!
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.correct, self.predictions)

        return ops, feed_dict

    def _variableSummaries(self, var):
        '''
        currently do not need any summaries in RNN implementation
        :param var:
        :return:
        '''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
