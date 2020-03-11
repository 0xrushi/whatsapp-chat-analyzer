#https://www.youtube.com/watch?v=37Zj955LFT0&list=PLQVvvaa0QuDfsGImWNt1eUEveHOepkjqt&index=4
import dash
from dash.dependencies import Output,Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
from constants import pattern_attachment, pattern_event, pattern_url

## Calculate year date range
from datetime import timedelta, date

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2016,1, 1)
end_dt = date(2020, 6, 3)
range_date=[]
for dt in daterange(start_dt, end_dt):
    range_date.append(dt.strftime("%Y-%m-%d"))

##

def top_words(df, x):
    popular_words={}
    global chat_words
    chat_words=''
    def get_words(msg):
        #remove non alpha content
        #print(msg)
        regex = re.sub(r"[^a-z\s]+", "", msg.lower())
        regex = re.sub(r'[^\x00-\x7f]',r'', regex)
        words = regex.split(" ")
        
        for x in words:
            if x:
                rank_word(x)
        return words
    
    def rank_word(word):
        common_words=['']
        if not word in common_words:
            popular_words[word] = popular_words.get(word, 0) + 1
            global chat_words
            chat_words += " {0}".format(word)
        return word
    
    temp= (df[df['member']== x])['message']
    for i in list(temp):
      for j in [pattern_attachment, pattern_event, [pattern_url]]:
        for p in j:
          i=re.sub(p, '', str(i))
      get_words(str(i))
    tem = pd.DataFrame(popular_words.items(), columns=("word", "count"))
    tem= tem.sort_values(by="count", ascending=True)
    return tem
###
    

person1_name= 'Vedang Pingle'
person2_name= '~/r4#51c0debl00d3D'

df= pd.read_csv("whatsappchat.csv")
df['Date2'] = pd.to_datetime(df['Date2'], errors='coerce')

temp= df.groupby(['Date2', 'member'], as_index=False).size().reset_index(name='messagecount')

person1_df= temp[temp['member']==person1_name]
person1_df.index = np.arange(len(person1_df))

person2_df= temp[temp['member']==person2_name]
person2_df.index = np.arange(len(person2_df))

person1_X=deque(maxlen=99999)
person1_X.append(1)
person11_X=df[1:1]

person1_Y=deque(maxlen= 99999)
person1_Y.append(1)
person2_X=deque(maxlen=99999)
person2_X.append(1)
person2_Y=deque(maxlen= 99999)
person2_Y.append(1)

i=0
ii=0
app=dash.Dash(__name__)
app.layout=html.Div([dcc.Graph(id='live-graph',animate=True),dcc.Interval(id='graph-update',interval=1000,n_intervals=0), 
                     dcc.Graph(id='live-graph2',animate=True),dcc.Interval(id='graph-update2',interval=1000,n_intervals=0), ])

@app.callback(Output('live-graph2','figure'),[Input('graph-update2','n_intervals')])
def update_graph_scatter2(n):
    global ii, range_date, person11_X
    print(ii)
    person11_X= person11_X.append(df[i:i+1])
    sorted_words= top_words(person11_X, 'Vedang Pingle').tail(10)
    person3_graph= plotly.graph_objs.Bar(x=list(sorted_words['word']),y=list(sorted_words['count']),name='Scatter')
    fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=False, vertical_spacing=0.009,horizontal_spacing=0.009)
    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
    fig.append_trace(person3_graph,1, 1)
    ii=ii+1
    return fig
    

@app.callback(Output('live-graph','figure'),[Input('graph-update','n_intervals')])
def update_graph_scatter(n):
    global i, range_date
    print(i)
    person1_X.append(person1_df['Date2'][i])
    person1_Y.append(person1_df['messagecount'][i])
    person2_X.append(person2_df['Date2'][i])
    person2_Y.append(person2_df['messagecount'][i])
    
    #data=plotly.graph_objs.Bar(x=list(X),y=list(Y),name='Scatter')
    person1_graph= plotly.graph_objs.Scatter(x=list(person1_X), y=list(person1_Y), name="PERSON1",
                         line_color='dimgray', mode= 'lines+markers')
    
    person2_graph= plotly.graph_objs.Scatter(x=list(person2_X), y=list(person2_Y), name="PERSON2",
                         line_color='red', mode= 'lines+markers')

    fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.009,horizontal_spacing=0.009)
    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
    fig.append_trace(person1_graph,1, 1)
    fig.append_trace(person2_graph,1, 1)
    
    
    i=i+1
    fig.layout= go.Layout(xaxis=dict(range=(min(pd.to_datetime(range_date)), max(pd.to_datetime(range_date)))),yaxis=dict(range=[0,max(person1_Y)]),)
    return fig
#    return{'data':[person1_graph],'layout':go.Layout(xaxis=dict(range=(min(pd.to_datetime(range_date)), max(pd.to_datetime(range_date)))),yaxis=dict(range=[0,max(person1_Y)]),)}

if __name__=='__main__':
    app.run_server(debug=True)
