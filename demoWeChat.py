from model.nnModel import ChatBot
import torch,warnings,itchat,argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='The path of your model file.', required=True, type=str)
parser.add_argument('--device', help='Your program running environment, "cpu" or "cuda"', type=str, default='cpu')
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    print('Loading the model...')
    chatBot = ChatBot(args.model, device=torch.device(args.device))
    print('Finished...')

    allRandomChoose,showInfo = False, False

    @itchat.msg_register(itchat.content.TEXT, isFriendChat=True)
    def text_reply(msg):
        global allRandomChoose, showInfo
        content = msg['Content']
        print('Friend:',content)
        if content=='_crazy_on_':
            allRandomChoose = True
            outputSeq = '成功开启疯狂模式...'
        elif content=='_crazy_off_':
            allRandomChoose = False
            outputSeq = '成功关闭疯狂模式...'
        elif content=='_showInfo_on_':
            showInfo = True
            outputSeq = '成功开启日志打印...'
        elif content=='_showInfo_off_':
            showInfo = False
            outputSeq = '成功关闭日志打印...'
        else:
            outputSeq = chatBot.predictByBeamSearch(content, isRandomChoose=True, allRandomChoose=allRandomChoose, showInfo=showInfo)
        print('AI:    ',outputSeq)
        print()
        return outputSeq

    itchat.auto_login()
    itchat.run()