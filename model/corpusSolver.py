import re,jieba,random,time
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Corpus:
    def __init__(self, filePath, maxSentenceWordsNum=-1, id2word=None, word2id=None, wordNum=None, tfidf=False, QIDF=None, AIDF=None, testSize=0.0):
        self.id2word, self.word2id, self.wordNum = id2word, word2id, wordNum
        with open(filePath,'r',encoding='utf8') as f:
            txt = self._purify(f.readlines())
            data = [i.split('\t') for i in txt]
            data = [[jieba.lcut(i[0]), jieba.lcut(i[1])] for i in data]
            data = [i for i in data if (len(i[0])<maxSentenceWordsNum and len(i[1])<maxSentenceWordsNum) or maxSentenceWordsNum==-1]
        self.chatDataWord = data
        self._word_id_map(data)
        try:
            chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
        except:
            chatDataId = [[[self.word2id[w] for w in qa[0] if w in self.id2word],[self.word2id[w] for w in qa[1] if w in self.id2word]] for qa in self.chatDataWord]
        self._QALens(chatDataId)
        self.maxSentLen = max(maxSentenceWordsNum, self.AMaxLen)
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
        self.totalSampleNum = len(data)
        if tfidf:
            #QTF = np.array([[qa[0].count(i)/len(qa[0]) for qa in chatDataId] for i in range(self.wordNum)], dtype='float32')
            #ATF = np.array([[qa[1].count(i)/len(qa[1]) for qa in chatDataId] for i in range(self.wordNum)], dtype='float32')
            self.QIDF = QIDF if QIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[0]) for qa in chatDataId])+1)) for i in range(self.wordNum)], dtype='float32')
            self.AIDF = AIDF if AIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[1]) for qa in chatDataId])+1)) for i in range(self.wordNum)], dtype='float32')
            #self.Q_TF_IDF = QTF*QIDF.reshape(-1,1)
            #self.A_TF_IDF = ATF*AIDF.reshape(-1,1)
        print("Total sample num:",self.totalSampleNum)
        self.trainIdList, self.testIdList = train_test_split([i for i in range(self.totalSampleNum)], test_size=testSize)
        self.trainSampleNum, self.testSampleNum = len(self.trainIdList), len(self.testIdList)
        print("train size: %d; test size: %d"%(self.trainSampleNum, self.testSampleNum))
        self.testSize = testSize
        print("Finished loading corpus!")
    def reset_word_id_map(self, id2word, word2id):
        self.id2word, self.word2id = id2word, word2id
        chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
    def random_batch_data_stream(self, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, type='train'):
        idList = self.trainIdList if type=='train' else self.testIdList
        eosToken, unkToken = self.word2id['<EOS>'], self.word2id['<UNK>']
        while True:
            samples = random.sample(idList, min(batchSize, len(idList))) if batchSize>0 else random.sample(idList, len(idList))
            if isDataEnhance:
                yield self._dataEnhance(samples, dataEnhanceRatio, eosToken, unkToken)
            else:
                QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
                QDataId = np.array([self.QChatDataId[i]+[eosToken for j in range(QMaxLen-self.QLens[i]+1)] for i in samples], dtype='int32')
                ADataId = np.array([self.AChatDataId[i]+[eosToken for j in range(AMaxLen-self.ALens[i]+1)] for i in samples], dtype='int32')
                yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]
    def one_epoch_data_stream(self, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, type='train'):
        idList = self.trainIdList if type=='train' else self.testIdList
        eosToken = self.word2id['<EOS>']
        for i in range(len(idList)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            if isDataEnhance:
                yield self._dataEnhance(samples, dataEnhanceRatio, eosToken, unkToken)
            else:
                QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
                QDataId = np.array([self.QChatDataId[i]+[eosToken for j in range(QMaxLen-self.QLens[i]+1)] for i in samples], dtype='int32')
                ADataId = np.array([self.AChatDataId[i]+[eosToken for j in range(AMaxLen-self.ALens[i]+1)] for i in samples], dtype='int32')
                yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]

    def _purify(self, txt):
        return [filter_sent(qa) for qa in txt]
    def _QALens(self, data):
        QLens, ALens = [len(qa[0])+1 for qa in data], [len(qa[1])+1 for qa in data]
        QMaxLen, AMaxLen = max(QLens), max(ALens)
        print('QMAXLEN:',QMaxLen,'  AMAXLEN:',AMaxLen)
        self.QLens, self.ALens = np.array(QLens, dtype='int32'), np.array(ALens, dtype='int32')
        self.QMaxLen, self.AMaxLen = QMaxLen, AMaxLen
    def _word_id_map(self, data):
        if self.id2word==None: 
            self.id2word = list(set([w for qa in data for sent in qa for w in sent]))
            self.id2word.sort()
            self.id2word = ['<EOS>','<SOS>'] + self.id2word + ['<UNK>']
        if self.word2id==None: self.word2id = {i[1]:i[0] for i in enumerate(self.id2word)}
        if self.wordNum==None: self.wordNum = len(self.id2word)
        print('Total words num:',len(self.id2word)-2)
    def _dataEnhance(self, samples, ratio, eosToken, unkToken):
        #print([[id2seq(self.id2word, self.QChatDataId[i]),id2seq(self.id2word, self.AChatDataId[i])] for i in samples][:1])    

        q_tf_idf = [[self.QIDF[w]*self.QChatDataId[i].count(w)/(self.QLens[i]-1) for w in self.QChatDataId[i]] for i in samples]
        #a_tf_idf = [[self.AChatDataId[i].count(w)/(self.ALens[i]-1) for w in self.AChatDataId[i]]for i in samples]

        q_tf_idf = [[j/sum(i) for j in i] for i in q_tf_idf]
        #a_tf_idf = [[j/sum(i) for j in i] for i in a_tf_idf]
        #print(q_tf_idf[:1])
        #print(a_tf_idf[:1])
        #print()
        QDataId = [[w for cntw,w in enumerate(self.QChatDataId[i]) if random.random()>ratio or random.random()<q_tf_idf[cnti][cntw]] for cnti,i in enumerate(samples)]
        QDataId = [[w if random.random()>ratio/5 else unkToken for w in self.QChatDataId[i]] for i in samples]
        #ADataId = [[w for cntw,w in enumerate(self.AChatDataId[i]) if random.random()>ratio or random.random()<a_tf_idf[cnti][cntw]] for cnti,i in enumerate(samples)]
        ADataId = [self.AChatDataId[i] for i in samples]

        for i in range(len(samples)):
            if random.random()<ratio:random.shuffle(QDataId[i])
            #   if random.random()<ratio:random.shuffle(ADataId[i])

        #print([[id2seq(self.id2word, QDataId[i]),id2seq(self.id2word, ADataId[i])] for i in range(len(samples))][:1])
        QLens = np.array([len(q)+1 for q in QDataId], dtype='int32')
        ALens = np.array([len(a)+1 for a in ADataId], dtype='int32')
        QMaxLen, AMaxLen = max(QLens), max(ALens)

        QDataId = np.array([q+[eosToken for j in range(QMaxLen-len(q))] for q in QDataId], dtype='int32')
        ADataId = np.array([a+[eosToken for j in range(AMaxLen-len(a))] for a in ADataId], dtype='int32')

        validId = (QLens>1) & (ALens>1)
        return QDataId[validId], QLens[validId], ADataId[validId], ALens[validId]

def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId
def id2seq(id2word, seqId):
    seqData = [id2word[i] for i in seqId]
    return seqData
def filter_sent(sent):
    return sent.replace('\n','').replace(' ','').replace('，',',').replace('。','.').replace('；',';').replace('：',':').replace('？','?').replace('！','!').replace('“','"').replace('”','"').replace("‘","'").replace("’","'").replace('（','(').replace('）',')')