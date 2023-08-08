import nagisa
from main import preprocessing

import codecs
import random
import re

from gpt3.api import GPT, Example
import openai

unk_token_word_list = []



lst = []
for unk in unk_token_word_list:
    for i in range(5):
        gpt = GPT(temperature=0.3, max_tokens=200)
        prompt = unk
        output = gpt.submit_request(prompt)
        lst.append(prompt + output['choices'][0]['text'][7:])
open('gpt3.txt', 'w', encoding='utf-8').write('\n'.join(lst))

preprocessing('gpt3.txt')

pretrained_hp = nagisa.utils.load_data("nagisa-kkma/nagisa_ko.hp")
pretrained_params = "nagisa-kkma/nagisa_ko.params"
pretrained_model = nagisa.model.Model(pretrained_hp, pretrained_params)
vocabs = nagisa.utils.load_data("nagisa-kkma/nagisa_ko.vocabs")

DELIMITER = "\t"
NEWLINE = "EOS"

train_data = nagisa.train.prepro.from_file(
    filename="nagisa-kkma/new_data.train",
    window_size=pretrained_hp['WINDOW_SIZE'],
    vocabs=vocabs,
    delimiter=DELIMITER,
    newline=NEWLINE
)

test_data = nagisa.train.prepro.from_file(
    filename="nagisa-kkma/new_data.test",
    window_size=pretrained_hp['WINDOW_SIZE'],
    vocabs=vocabs,
    delimiter=DELIMITER,
    newline=NEWLINE
)

dev_data = nagisa.train.prepro.from_file(
    filename="nagisa-kkma/new_data.dev",
    window_size=pretrained_hp['WINDOW_SIZE'],
    vocabs=vocabs,
    delimiter=DELIMITER,
    newline=NEWLINE
)

retrained_model_name = "nagisa_ko_new"
pretrained_hp["MODEL"] = f"{retrained_model_name}.params"
pretrained_hp["HYPERPARAMS"] = f"{retrained_model_name}.hp"

nagisa.train._start(pretrained_hp, pretrained_model, train_data, test_data, dev_data)




# tagger = nagisa.Tagger(
#     vocabs='./nagisa-kkma/nagisa_ko.vocabs',
#     params='./nagisa-kkma/nagisa_ko.params',
#     hp='./nagisa-kkma/nagisa_ko.hp'
# )
# print('\n형태소 분석기 kkma, nagisa_ko 로드 완료\n')

# text = open('time_test.txt', 'r', encoding='utf-8').read()


# words = str(tagger.tagging(text))
# print(words)
