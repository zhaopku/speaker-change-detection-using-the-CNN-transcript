import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import argparse
from speakerChangeTagger.textData import TextData
from speakerChangeTagger.RecurrentModel import RecurrentModel
from sklearn.metrics import confusion_matrix
import pickle as p

class Tagger:
    def __init__(self):
        self.args = None

        self.textData = None
        self.model = None

        self.globalStep = 0

        self.summaryWriter = None
        self.outFile = None
        self.mergedSummary = None
        # tensorflow main session
        self.sess = None

    @staticmethod
    def parseArgs(args):

        parser = argparse.ArgumentParser()

        parser.add_argument('--resultDir', type=str, default='result', help='result directory')

        # data location
        dataArgs = parser.add_argument_group('Dataset options')

        dataArgs.add_argument('--trainData', type=str, default='data/train', help='training data location')
        dataArgs.add_argument('--valData', type=str, default='data/val', help='validation data location')
        dataArgs.add_argument('--testData', type=str, default='data/test', help='test data location')
        dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
        dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')
        dataArgs.add_argument('--numClasses', type=int, default=2, help='number of classes for current dataset')
        dataArgs.add_argument('--summaryDir', type=str, default='summaries', help='directory of summaries')
        dataArgs.add_argument('--minOccur', type=int, default=5, help='min occurances for a word')

        # CNN network options
        nnArgs = parser.add_argument_group('Network options')
        nnArgs.add_argument('--maxLength', type=int, default=30, help='maximum length for one utterance, useful for padding')
        nnArgs.add_argument('--wordWindowSize', type=int, default=5, help='CNN window size for words')
        nnArgs.add_argument('--wordCNNSize', type=int, default=200, help='CNN size for words')
        nnArgs.add_argument('--uttWindowSize', type=int, default=3, help='CNN windows size for utterances')
        nnArgs.add_argument('--uttCNNSize', type=int, default=200, help='CNN size for utterance')
        nnArgs.add_argument('--embeddingSize', type=int, default=200, help='embedding size')

        # RNN network options
        rnnArgs = parser.add_argument_group('RNN options')
        rnnArgs.add_argument('--wordUnits', type=int, default=200)
        rnnArgs.add_argument('--wordLayers', type=int, default=1)
        rnnArgs.add_argument('--uttUnits', type=int, default=300)
        rnnArgs.add_argument('--uttLayers', type=int, default=1)

        # training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--dropOut', type=float, default=0.9, help='dropout rate for CNN')
        trainingArgs.add_argument('--learningRate', type=float, default=0.0015, help='learning rate')
        trainingArgs.add_argument('--batchSize', type=int, default=50, help='batch size')
        ## do not add dropOut in the test mode!
        trainingArgs.add_argument('--test', type=bool, default=False, help='if in test mode')
        trainingArgs.add_argument('--epochs', type=int, default=100, help='training epochs')
        trainingArgs.add_argument('--device', type=str, default='/gpu:1', help='use the second GPU as default')
        trainingArgs.add_argument('--preEmbedding', type=bool, default=False, help='whether or not to use the pretrained embedding')
        trainingArgs.add_argument('--embeddingFile', type=str, default='embeddings/embedding-200.pkl', help='pretrained embeddings')
        return parser.parse_args(args)


    def constructFileName(self):
        file_name = str(self.args.maxLength) + '_' + str(self.args.wordWindowSize) + '_' + str(self.args.wordCNNSize)
        file_name += '_' + str(self.args.uttWindowSize) + '_' + str(self.args.uttCNNSize) + '_'
        file_name += str(self.args.embeddingSize) + '_' + str(self.args.dropOut) + '_' + str(self.args.learningRate)
        file_name += '_' + str(self.args.batchSize)

        return file_name

    def constructDatasetName(self):
        suffix = '-' + str(self.args.maxLength) + '-' + str(self.args.uttWindowSize) + '-' + str(self.args.batchSize)
        self.args.datasetName += suffix + '.pkl'

    def constructDir(self, base):
        directory = []
        directory.append(base)
        # maxlength
        super = []
        super.append('maxLen_' + str(self.args.maxLength))
        super.append('wWin_' + str(self.args.wordWindowSize))
        super.append('wCNN_' + str(self.args.wordCNNSize))
        super.append('uWin_' + str(self.args.uttWindowSize))
        super.append('uCNN_' + str(self.args.uttCNNSize))
        super.append('e_' + str(self.args.embeddingSize))

        base_0 = ''.join(super)

        dir_0 = os.path.join(base, base_0)
        if not os.path.exists(dir_0):
            os.mkdir(dir_0)

        dir_1 = os.path.join(dir_0, 'lr_' + str(self.args.learningRate))
        if not os.path.exists(dir_1):
            os.mkdir(dir_1)

        dir_2 = os.path.join(dir_1, 'dropout_' + str(self.args.dropOut))
        if not os.path.exists(dir_2):
            os.mkdir(dir_2)
        return dir_2



    def main(self, args=None):
        print('TensorFlow v{}'.format(tf.__version__))

        # initialize args
        self.args = self.parseArgs(args)

        self.outFile = self.constructFileName()

        # note: for padding
        assert self.args.uttWindowSize < self.args.batchSize

        # load data if exists, else create the dataset
        self.constructDatasetName()
        datasetFileName = os.path.join(self.args.dataDir, self.args.datasetName)
        if not os.path.exists(datasetFileName):
            self.textData = TextData(self.args)
            with open(datasetFileName, 'wb') as datasetFile:
                p.dump(self.textData, datasetFile)
            print('dataset created and saved to {}'.format(datasetFileName))
        else:
            with open(datasetFileName, 'rb') as datasetFile:
                self.textData = p.load(datasetFile)
            print('dataset loaded from {}'.format(datasetFileName))


        # note: since dropOut is not implemented yet, currently we only have one model

        # default session
        sessConfig = tf.ConfigProto(allow_soft_placement=True)
        sessConfig.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sessConfig)

        # summary writer
        self.args.summaryDir = self.constructDir('summaries')
        with tf.device(self.args.device):
            self.model = RecurrentModel(self.args, self.textData)
            self.summaryWriter = tf.summary.FileWriter(self.args.summaryDir, self.sess.graph)
            self.mergedSummary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            # initialize all global variables
            self.sess.run(init)
            self.train(self.sess)

    def train(self, sess):
        '''
        training loop
        :param sess: default sess for Tagger
        :return:
        '''

        print('Starting training')

        out = open(os.path.join(self.args.resultDir, self.outFile), 'w', 1)
        out.write(self.outFile + '\n')

        for e in range(self.args.epochs):
            # training
            trainBatches = self.textData.getBatches('train')
            totalTrainLoss = 0.0
            allTrainCorrect = 0.0
            TP = 0
            TN = 0
            FP = 0
            FN = 0

            for nextBatch in tqdm(trainBatches):
            #for idx, nextBatch in enumerate(trainBatches):
                self.globalStep += 1
                '''
                seqNum = len(nextBatch.sequence)
                lenLen = len(nextBatch.length)
                assert seqNum == lenLen
                if seqNum != lenLen:
                    print('sequence length = {}, length = {}'.format(seqNum, lenLen))

                #print(nextBatch.id)
                continue
                '''
                ops, feed_dict = self.model.step(nextBatch)
                ops = (self.mergedSummary,) + ops

                # use vec to fetch some vector from the graph, just for debug
                batchSummary, _, loss, correct, predictions, vec = sess.run(ops, feed_dict)

                #self.summaryWriter.add_summary(batchSummary, self.globalStep)

                totalTrainLoss += loss
                allTrainCorrect += correct
                true_positive, true_negative, false_postive, false_negative \
                    = self.calculate_F1(predictions, nextBatch.labels)
                TP += true_positive
                TN += true_negative
                FP += false_postive
                FN += false_negative

            precision = (TP*1.0)/(TP+FP)
            recall = (TP*1.0)/(TP+FN)

            f1 = 2*precision*recall/(precision+recall)

            trainAcc = allTrainCorrect/self.textData.trainCnt
            valAcc, valF1, valP, valR = self.test(sess, tag='val')
            testAcc, testF1, testP, testR = self.test(sess, tag='test')

            tf.summary.scalar(name='trainF1', tensor=f1)
            tf.summary.scalar(name='valF1', tensor=valF1)
            tf.summary.scalar(name='testF1', tensor=testF1)
            tf.summary.scalar(name='trainLoss', tensor=totalTrainLoss)
            tf.summary.scalar(name='trainAcc', tensor=trainAcc)
            tf.summary.scalar(name='valAcc', tensor=valAcc)
            tf.summary.scalar(name='testAcc', tensor=testAcc)

            print('epoch = {}/{}, trainAcc = {}, trainLoss = {}, valAcc = {}, testAcc = {}'
                  .format(e+1, self.args.epochs, trainAcc, totalTrainLoss, valAcc, testAcc))
            print('trainF1 = {}, valF1 = {}, testF1 = {}'.format(f1, valF1, testF1))
            print('trainP = {}, trainR = {}, valP = {}, valR = {}, testP = {}, testR = {}'.format(precision, recall,
                                                                                                  valP, valR, testP, testR))
            out.write('epoch = {}/{}, trainAcc = {}, trainLoss = {}, valAcc = {}, testAcc = {}\n'
                  .format(e+1, self.args.epochs, trainAcc, totalTrainLoss, valAcc, testAcc))
            out.write('               trainF1 = {}, valF1 = {}, testF1 = {}\n'
                      .format(f1, valF1, testF1))
            out.write('               trainP = {}, trainR = {}, valP = {}, valR = {}, testP = {}, testR = {}\n'
                      .format(precision, recall, valP, valR, testP, testR))
            out.flush()
        out.close()


    def test(self, sess, tag='val'):
        batches = self.textData.getBatches(tag)
        allCorrect = 0.0

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for idx, nextBatch in enumerate(batches):
            ops, feed_dict = self.model.step(nextBatch, test=True)
            correct, predictions = sess.run(ops, feed_dict)
            true_positive, true_negative, false_postive, false_negative \
                = self.calculate_F1(predictions, nextBatch.labels)

            allCorrect += correct
            TP += true_positive
            TN += true_negative
            FP += false_postive
            FN += false_negative

        if TP == 0 and FP == 0:
            f1 = -1.0
            precision = (TP * 1.0) / (TP + FP)
            recall = (TP * 1.0) / (TP + FN)
        else:
            precision = (TP * 1.0) / (TP + FP)
            recall = (TP * 1.0) / (TP + FN)

            f1 = 2 * precision * recall/(precision + recall)

        if tag == 'val':
            acc = allCorrect / self.textData.valCnt
        else:
            acc = allCorrect / self.textData.testCnt

        return acc, f1, precision, recall


    def calculate_F1(self, predictions, labels):
        '''
        calculate TP, FP, FN
        :return:
        '''
        # sometimes predictions and labels are all zeros or all ones, we need *labels* to guide
        matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0,1])

        true_positive = matrix[1][1]
        false_postive = matrix[0][1]

        true_negative = matrix[0][0]
        false_negative = matrix[1][0]

        return true_positive, true_negative, false_postive, false_negative

