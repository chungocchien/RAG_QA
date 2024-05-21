import pandas as pd
import numpy as np

f = open("cau_tra_loi_1.txt", "r", encoding='utf-8')
s = f.read()
chunks = s.split('<START>')
chunks.remove('')

with open('answer_1.txt', 'r', encoding='utf-8') as file:
    real = file.readlines()
import evaluate

bleu = evaluate.load('bleu')
results = bleu.compute(predictions=chunks, references=real)
print(results)