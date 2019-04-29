from model.nnModel import ChatBot
import torch,warnings
warnings.filterwarnings("ignore")

print('Loading the model...')
chatBot = ChatBot('model_L_general_all.pkl', device=torch.device('cuda:0'))
print('Finished...')

allRandomChoose,showInfo = False, False
while True:
    inputSeq = input("You: ")
    if inputSeq=='_crazy_on_':
        allRandomChoose = True
        print('AI: ','成功开启疯狂模式...')
    elif inputSeq=='_crazy_off_':
        allRandomChoose = False
        print('AI: ','成功关闭疯狂模式...')
    elif inputSeq=='_showInfo_on_':
        showInfo = True
        print('AI: ','成功开启日志打印...')
    elif inputSeq=='_showInfo_off_':
        showInfo = False
        print('AI: ','成功关闭日志打印...')
    else:
        outputSeq = chatBot.predictByBeamSearch(inputSeq, isRandomChoose=True, allRandomChoose=allRandomChoose, showInfo=showInfo)
        print('AI: ',outputSeq)
    print()