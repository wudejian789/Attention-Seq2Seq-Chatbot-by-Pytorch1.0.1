Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1
===
![totalModel](https://github.com/wudejian789/Attention-Seq2Seq-Chatbot-by-Pytorch1.0.1/blob/master/image/totalModel.png)
>Note: If there's any error like "No module named 'xxx'", please use command "pip install xxx" to repair.

# 1. Import the module
```python
from model.nnModel import *
from model.corpusSolver import *
import torch
```
# 2. How to load the data
```python
dataClass = Corpus('./corpus/qingyun.tsv', maxSentenceWordsNum=25)
```
>***First parameter*** is your corpus path;  
>***maxSentenceWordsNum*** will ignore the data whose words number of question or answer is too big;  

Also you can load your corpus. Only your file content formats need to be consistent:
>Q  A  
>Q  A  
>...

Every line is a question and a answer with a '\t' split.
# 3. How to train your model
First you need create a Seq2Seq object.
```python
model = Seq2Seq(dataClass, featureSize=256, hiddenSize=256, 
                attnType='L', attnMethod='general', 
                encoderNumLayers=3, decoderNumLayers=2, 
                encoderBidirectional=True, 
                device=torch.device('cuda:0'))
```
>***First parameter*** is your corpus class object.  
>***featureSize*** is your word vector size;  
>***hiddenSize*** is your RNN hidden state size;  
>***attnType*** is your attention type. It can be 'B' for using Bahdanau Attention Structure or 'L' for using Luong Structure;  
> ***attnMethod*** is Luong Attention Method. It can be 'dot', 'general' or 'concat'.  
>***encoderNumLayers*** is the layer number of your encoder RNN;  
>***decoderNumlayers*** is the layer number of your decoder RNN;  
>***encoderBidirectional*** is if your encoder RNN is bidirectional;  
>***device*** is your building environment. If using CPU, then device=torch.device('cpu'); if using GPU, then device=torch.device('cuda:0');  

Then you can train your model.
```python
model.train(batchSize=1024, epoch=500)
```
>***batchSize*** is the number of data used for each train step;  
>***epoch*** is the total iteration number of your training data;  

And the log will be print like follows:
```
...
After iters 6540: loss = 0.844; train bleu: 0.746, embAve: 0.754; 2034.582 qa/s; remaining time: 48096.110s;
After iters 6550: loss = 0.951; train bleu: 0.734, embAve: 0.755; 2034.518 qa/s; remaining time: 48092.589s;
After iters 6560: loss = 1.394; train bleu: 0.735, embAve: 0.759; 2034.494 qa/s; remaining time: 48088.128s;
...
```
Finally you need to save your model for future use.
```
model.save('model.pkl')
```
>First parameter is the name of model saved.  

Ok, I know you are too lazy to train your own model. Also, you can download my trained model from my Baidu Net Disk.  
The model below is trained in qingyun corpus.  
|encoder|decoder|attention|data enhance|test size|address|key|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|5×Bi_GRU|3×GRU|Bahdanau(concat)|False|0.1|[link](https://pan.baidu.com/s/1qel4uPNAdVF7Sjl-fzWAuQ)|s55l|
|5×Bi_GRU|3×GRU|Luong(dot)|False|0.1|[link](https://pan.baidu.com/s/1ftVs682QzmFDqPRdSgN7Zg)|x76r|
|5×Bi_GRU|3×GRU|Luong(general)|False|0.1|[link](https://pan.baidu.com/s/1uVg4IwnPzCx7H48wFmjWOA)|p3y0|
|5×Bi_GRU|3×GRU|Luong(concat)|False|0.1|[link](https://pan.baidu.com/s/16SnTTx8CQBhnkEOe6Dj0QA)|xte1|
|5×Bi_GRU|3×GRU|Luong(general)|False|0.0|[link](https://pan.baidu.com/s/1pn4_6JCco95g9JHxC0R9FQ)|pl5j|
|5×Bi_GRU|3×GRU|Luong(general)|True|0.0|[link](https://pan.baidu.com/s/1_GHEDRzQyl-R5LIndgQurQ)|0sfe|
# 4. How to use your model to build a chatbot
First you need create a Chatbot object.
```python
chatbot = Chatbot('model.pkl')
```
>First parameter is your model path;  

Then you can use the greedy search to generate the answer.
```python
chatbot.predictByGreedySearch("你好啊")
```
>First parameter is your question;  

It will return the answer like "你好,我就开心了". Also you can plot the attention by ***showAttention=True***.
Or you can use the beam search to generate the answer.
```python
chatbot.predictByBeamSearch("什么是ai", isRandomChoose=True, beamWidth=10)
```
>***First parameter*** is your question;  
>***isRandomChoose*** determines whether probability sampling is performed in the final beamwidth answers.  
>***beamWidth*** is the search width in beam search;   

It will return the answer like "反正不是苹果". Also you can show the probabilities of the beamwidth answers by ***showInfo=True***.
# 5. Other function
For other functions, please dig for yourselves.
