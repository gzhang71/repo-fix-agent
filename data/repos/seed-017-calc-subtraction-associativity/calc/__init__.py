from .evaluator import evaluate
from .parser import Parser
from .tokenizer import tokenize


def calculate(text):
    return evaluate(Parser(tokenize(text)).parse())
