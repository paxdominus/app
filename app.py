#import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# initialize tokenizer and model from pretrained GPT2 model


from flask import jsonify, make_response
from flask import Flask, request
from markupsafe import Markup

app = Flask(__name__)


@app.route('/')
def home():
    return 'Home Page Route'


@app.route('/data')
def about():
        sequence =  request.args.get('data')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        inputs = tokenizer.encode(sequence, return_tensors='pt')
        outputs = model.generate(inputs, max_length=200, do_sample=True)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("aa"+text)
        d = {"data": text}
        return make_response(jsonify(d), 200)


@app.route('/portfolio')
def portfolio():
    return 'Portfolio Page Route'


@app.route('/contact')
def contact():
    return 'Contact Page Route'


@app.route('/api')
def api():
    with open('data.json', mode='r') as my_file:
        text = my_file.read()
        return text

if __name__ == '__main__':
	app.run(port=80)
