

def init():
    global QUERY
    QUERY = ""

def add_query(query):
    global QUERY
    QUERY = query

def get_query():
    global QUERY
    return QUERY

def destroy():
    global QUERY
    QUERY = ""