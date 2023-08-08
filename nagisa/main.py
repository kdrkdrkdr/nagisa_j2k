import os
from time import time
from konlpy.tag import Kkma
import nagisa

tagger = Kkma()

def preprocessing(dataset):
    lst = []
    f = open(dataset, 'r', encoding='utf-8').read().split('\n')

    for c, i in enumerate(f):
        print(c, i)
        p = tagger.pos(i)
        for j in p:
            lst.append(f'{j[0]}\t{j[1]}')
        lst.append('EOS')
        
    data = '\n'.join(lst)

    open(f'kkma.{dataset.split(".")[1]}', 'w', encoding='utf-8').write(data)



if __name__ == "__main__":
    # Start Preprocessing
    # preprocessing('train.txt')

    # # Start Training 
    # os.chdir('nagisa-kkma')
    # nagisa.fit(
    #     train_file='../kkma.train',
    #     dev_file='../kkma.dev',
    #     test_file='../kkma.test',
    #     model_name='nagisa_ko'
    # )

    kkma = Kkma()
    tagger = nagisa.Tagger(
        vocabs='./nagisa-kkma/nagisa_ko.vocabs',
        params='./nagisa-kkma/nagisa_ko.params',
        hp='./nagisa-kkma/nagisa_ko.hp'
    )
    print('\n형태소 분석기 kkma, nagisa_ko 로드 완료\n')

    text = open('time_test.txt', 'r', encoding='utf-8').read()

    start = time()
    words = str(kkma.pos(text))
    end = (time() - start) 
    print('kkma:', end, '초')

    start = time()
    words = str(tagger.tagging(text))
    end = (time() - start) 
    print('nagisa_ko:', end, '초')
    
    