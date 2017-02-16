'''
A convolutional neural network for speaker change detection
Created by Zhao Meng, zhaomeng at pku.edu.cn
'''
import tensorflow as tf

class ConvolutionModel:
    def __init__(self, args, textData):
        self.args = args
        self.textData = textData

        self.input_utterances = None
        self.labels = None
        self.embedded = None

        # add a channel dimension to the embedded, in this case, channel = 1
        self.embedded_expand = None
        # Warning: do not access self.args.batchSize directly

        self.predictions = None

        self.loss = None
        self.correct = None
        self.accuracy = None
        self.correct_predictions = None

        # this dropout rate is just a placeholder, do not access self.args.dropOut directly!
        self.dropOutRate = None

        self.buildNetwork()

    def buildNetwork(self):
        pool_squeeze = self._buildWordNetwork()

        vectors =  self._buildUttNetwork(pool_squeeze)

        with tf.name_scope('output'):

            weights = tf.Variable(tf.truncated_normal([self.args.uttCNNSize, self.args.numClasses], stddev=0.5),
                                  name='weights')
            self._variableSummaries(weights)
            biases = tf.Variable(tf.truncated_normal([self.args.numClasses], stddev=0.5), name='biases')
            self._variableSummaries(biases)
            logits = tf.add(tf.matmul(vectors, weights), biases, name='logits')
            self._variableSummaries(logits)
            # the shape of output is like this: [?, 2], ? is the batch size, 2 is number of classes
            #output = tf.nn.softmax(logits, name='softmax')

            # because we use next batch's sentence to padding
            # so we will have more sentences than we need

            # at the end of a show, self.batchSize may not equal to self.args.batchSize
            # we use additional utterances to pad, remove them now
            # note
            self.out = tf.strided_slice(logits, begin=[self.args.uttWindowSize-1, 0], end=[-self.args.uttWindowSize+1, self.args.numClasses],
                                   strides=[1, 1], name='softmax_truncated')
            # dimension 0 is different samples in the batch, hence reduce dimension 1, which is the probability
            self.predictions = tf.argmax(self.out, axis=1, name='predictions')
            #self._variableSummaries(self.predictions)
        with tf.name_scope('loss'):
            # note: since we have the sparse version, we don't need to have one-hot labels
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(self.out, self.labels, name='loss_')
            self.loss = tf.reduce_mean(loss_, name='loss')
            self._variableSummaries(loss_)
            self._variableSummaries(self.loss)
        with tf.name_scope('evaluation'):
            self.correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)

            self.correct = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32), name='numCorrect')
            self._variableSummaries(self.correct)

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')
            self._variableSummaries(self.accuracy)

        with tf.name_scope('backpropagation'):
            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.minimize(self.loss)


    def _buildWordNetwork(self):
        # input for utterances, all the utterances of a show is input at the same time
        # here, 'None' indicates each show has different number of utterances, which is treated as batch size later
        # in the first CNN, pay attention that for the second CNN, batch size has different meaning

        # due to the batching mechanism, input_utterances's first dimension > labels.first_dimension
        self.input_utterances = tf.placeholder(tf.int32, [None, self.args.maxLength], name='input_utterances')
        # note: for each input utterances, we do not need one-hot vector, shape = [true_batch_size]
        # note: we have two placeholder in the network
        self.labels = tf.placeholder(tf.int32, None, name='labels')

        # note, this drop out is a place holder!
        self.dropOutRate = tf.placeholder(tf.float32, None, name='dropOut')

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
            # self.embedded is a 3-dimentional matrix of shape [None, maxLength, embeddingSize]
            self.embedded = tf.nn.embedding_lookup(embeddings, self.input_utterances)
            self.vectors = embeddings
            self._variableSummaries(self.embedded)
            # self.embedded_expand is a 4-dimensional matrix of shape of [None, maxLength, embeddingSize, 1]
            # the last dimension is channel, which, in this case, is 1
            self.embedded_expand = tf.expand_dims(self.embedded, -1)
            self._variableSummaries(self.embedded_expand)


        # convolution and max-pooling
        # in this problem, only has 1 filter
        with tf.name_scope('word_conv_maxPool'):
            # input should be: [batch, in_height, in_width, in_channels]
            # [filter_height, filter_width, in_channels, out_channels]
            # filter_height means window size, filter_width is embedding size, out_channels is the dimension of the resulting vector
            filterShape = [self.args.wordWindowSize, self.args.embeddingSize, 1, self.args.wordCNNSize]

            weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.5), name='weights')
            self._variableSummaries(weights)

            biases = tf.Variable(tf.truncated_normal([self.args.wordCNNSize], stddev=0.5), name='biases')
            self._variableSummaries(biases)
            # conv is the collection of all the resulting vectors, add biases to each of them
            # use padding, padding would be unabled if we set padding='VALID'
            # [1, height, width, 1], height is the stride of words
            conv = tf.nn.conv2d(self.embedded_expand, weights, [1, 1, self.args.embeddingSize, 1], padding='VALID', name='conv')

            # during test, do not use drop out

            self._variableSummaries(conv)
            # remove dimensions if size 1
            # this makes conv like this: [numVectors, out_channels]
            #conv = tf.squeeze(conv)

            # add biases, which would be broadcast to each resulting vector automatically
            conv_biases = tf.add(conv, biases, name = 'conv_biases')
            self._variableSummaries(conv_biases)
            conv_activated = tf.nn.relu(conv_biases, name='relu')

            # note dropout
            with tf.name_scope('drop_out'):
                conv_activated = tf.nn.dropout(conv_activated, self.dropOutRate, name='conv_drop')

            self._variableSummaries(conv_activated)
            shapeConv = conv_activated.get_shape()

            # do pooling over diffrent output vectors of the conv network
            # pool should have this shape: [batch_size, 1, 1, out_channels]
            # up till here, batch size is the number of all the sentences in a show
            pool = tf.nn.max_pool(conv_activated, ksize=[1, shapeConv[1], 1, 1], strides=[1, 1, 1, 1],
                                    name='pool', padding='VALID')
            self._variableSummaries(pool)
            pool_squeeze = tf.squeeze(pool, [1, 2], name='pool_squeeze')
            self._variableSummaries(pool_squeeze)
            return pool_squeeze


    def _buildUttNetwork(self, pool_squeeze):

        with tf.name_scope('utt_conv_maxPool'):
            # pool_squeeze has size like this: [num of utterences in a show, length of a vector]
            pool_squeeze_expand = tf.expand_dims(pool_squeeze, 0)
            self._variableSummaries(pool_squeeze_expand)
            # [batch_size = 1, number of utterences in a show, the dimension of the pooled vector, 1]
            # the dimension of the pooled vector = self.arg.wordCNNSize
            pool_squeeze_expand = tf.expand_dims(pool_squeeze_expand, -1, name='pool_squeeze_expand')
            filterShape = [self.args.uttWindowSize, self.args.wordCNNSize, 1, self.args.uttCNNSize]

            weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.5), name='weights')
            self._variableSummaries(weights)
            biases = tf.Variable(tf.truncated_normal([self.args.uttCNNSize], stddev=0.5), name='biases')
            self._variableSummaries(biases)

            conv = tf.nn.conv2d(pool_squeeze_expand, weights, [1, 1, self.args.uttCNNSize, 1], padding='SAME',
                                name='conv')

            # during test, do not use drop out

            self._variableSummaries(conv)
            # add biases, which would be broadcast to each resulting vector automatically
            conv_biases = tf.add(conv, biases, name='conv_biases')
            self._variableSummaries(conv_biases)
            conv_activated = tf.nn.relu(conv_biases, name='relu')

            # note dropout
            with tf.name_scope('drop_out'):
                conv_activated = tf.nn.dropout(conv_activated, self.dropOutRate, name='conv_drop')
              #  self._variableSummaries(conv_activated)
            self._variableSummaries(conv_activated)
            # conv_squeezed has shape like this: [?, self.args.uttCNNSize]
            # each vector in ? should be treated as input to softmax
            conv_squeezed = tf.squeeze(conv_activated, [0, 2], name='conv_squeezed')
            self._variableSummaries(conv_squeezed)
            return conv_squeezed


    def _variableSummaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def step(self, batch, test = False):
        feed_dict = {}
        ops = None

        feed_dict[self.input_utterances] = batch.sequence
        feed_dict[self.labels] = batch.labels

        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.correct, self.predictions, self.vectors)
        else:
            # during test, do not use drop out!!!!
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.correct, self.predictions)

        return ops, feed_dict