import sys
from transformers import pipeline
from transformers import AutoTokenizer

def main():
    user_input = ''
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    emojify = pipeline("text-classification", model="./model", tokenizer=tokenizer)
    tokenizer_kwargs = {'padding':True,'truncation':True}
    while(user_input != "quit"):
        user_input = input("Enter the message you would like to emojify!\n")
        prediction = emojify([user_input], **tokenizer_kwargs)
        print("Your prediction is: {}".format(prediction))

if __name__=='__main__':
    main()