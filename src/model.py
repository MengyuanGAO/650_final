import main
import flask
import pandas as pd
import tensorflow as tf

def init():
    global QUERY, MODEL
    global graph
    graph = tf.get_default_graph()
    MODEL = main.modelInit()
    QUERY = ""

def add_query(query):
    global QUERY
    QUERY = query

def get_query():
    global QUERY
    return QUERY

def predicting():
    global QUERY, MODEL, graph
    with graph.as_default():
        if QUERY == "":
            QUERY = "Why don't poor countries print more money to use for paying for education, etc.?"
            print(QUERY)
        ans = str(main.queryPredict(QUERY, MODEL))
    return ans

def destroy():
    global QUERY
    QUERY = ""