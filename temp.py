import requests
from typing import List
import torch

response=requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt") 


text=response.text


##### tiktoken

# This also can be used to tokenize. I mean we don't have to worry about tokenization stuff

# ```py
# import tiktoken

# enc = tiktoken.encoding_for_model("gpt-4o") # this download the encoding used for gpt-4o model
# enc.encode("hello world ")
# ```


char_to_index={char:index for index,char in enumerate(sorted(set(text)))}
index_to_char={index:char for index,char in enumerate(sorted(set(text)))}
print(index_to_char[2])
print(char_to_index['!'])



word='hello world'
def encode(sent:str)->List[int]:
    outpur_seq=[]
    for char in sent:
        outpur_seq.append(char_to_index[char])
    return outpur_seq
    

def decode(seq:List[int])->str:
    output_seq=""
    for index in seq:
        output_seq+=index_to_char[index]
    
    return output_seq

print(encode(word))
print(decode([46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]))

# encode the text
data = torch.tensor(encode(text), dtype=torch.long)