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
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
    

## get top emojis
import emoji

def get_top_emoji(df):
    ar=[]
    for i in df['emoji']:
      for j in i:
        if j in emoji.UNICODE_EMOJI:
          ar.append(j)
    
    def increment_emoji_count(char_list):
        emojis={"emj_char":[], "char_count":[]}
        groups = Counter(char_list)
        for c in groups.items():
            emojis["emj_char"].append(c[0])
            emojis["char_count"].append(c[1])
        return pd.DataFrame(emojis).sort_values(by="char_count", ascending=False)
    return increment_emoji_count(ar)

##    
    

person1_name= 'Vedang Pingle'
person2_name= '~/r4#51c0debl00d3D'

df= pd.read_csv("whatsappchat.csv")
df['Date2'] = pd.to_datetime(df['Date2'], errors='coerce')

temp= df.groupby(['Date2', 'member'], as_index=False).size().reset_index(name='messagecount')

person1_df= temp[temp['member']==person1_name]
person1_df.index = np.arange(len(person1_df))
person1_df['weekday']=person1_df['Date2'].dt.weekday_name
person1_radar_chart_weekly_df= person1_df.groupby(by='weekday')['messagecount'].agg('sum').to_frame().reset_index()
person1_message_count= person1_df['messagecount'].sum()

person1_messages= ''.join(df[df['member']==person1_name]['message'].apply(str))
person1_messages= re.sub(r"(<Media omitted>|[^a-zA-Z0-9]+)"," ", person1_messages)
person1_word_count= len(person1_messages.split())

person2_df= temp[temp['member']==person2_name]
person2_df.index = np.arange(len(person2_df))
person2_df['weekday']=person2_df['Date2'].dt.weekday_name
person2_radar_chart_weekly_df= person2_df.groupby(by='weekday')['messagecount'].agg('sum').to_frame().reset_index()
person2_message_count= person2_df['messagecount'].sum()
person2_messages= ''.join(df[df['member']==person2_name]['message'].apply(str))
person2_messages= re.sub(r"(<Media omitted>|[^a-zA-Z0-9]+)"," ", person2_messages)
person2_word_count= len(person2_messages.split())


## word count plots
#fig = px.pie(pd.DataFrame({'person':[person1_name, person2_name], 'count':[person1_word_count, person2_word_count]}),
#             values='count', names='person', color='person',
#             color_discrete_map={person1_name:'lightcyan', 
#                                 person2_name:'cyan', 
#                                 'Sat':'royalblue', 
#                                 'Sun':'darkblue'})
#
#fig.show()
##

## hourly radar chart stuff
temp= df.groupby(['Time', 'member'], as_index=False).size().reset_index(name='messagecount')
temp['Time'] = pd.to_datetime(temp['Time'], errors='coerce')
temp['hoours']= temp['Time'].dt.hour
temp= temp.groupby(['hoours', 'member'], as_index=False).size().reset_index(name='messagecount')
person1_radar_chart_hourly_df= temp[temp['member']== person1_name]
person2_radar_chart_hourly_df= temp[temp['member']== person2_name]

person1_radar_chart_hourly_df['hoours']=person1_radar_chart_hourly_df['hoours'].apply(str)
person2_radar_chart_hourly_df['hoours']=person2_radar_chart_hourly_df['hoours'].apply(str)
person1_radar_chart_hourly_df['hoours']= 'hour_'+person1_radar_chart_hourly_df['hoours']
person2_radar_chart_hourly_df['hoours']= 'hour_'+person2_radar_chart_hourly_df['hoours']
person1_radar_chart_hourly_df=person1_radar_chart_hourly_df.reset_index()

chat_words=''


## Emoji wordcloud
import string
import os, base64
from os import path
normal_word = r"(?:\w[\w']+)"
# 2+ consecutive punctuations, e.x. :)
ascii_art = r"(?:[{punctuation}][{punctuation}]+)".format(punctuation=string.punctuation)
# a single character that is not alpha_numeric or other ascii printable
emojee = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)
regexp = r"{normal_word}|{ascii_art}|{emoji}".format(normal_word=normal_word, ascii_art=ascii_art, emoji=emojee)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
font_path = path.join(d, 'fonts', 'Symbola', 'Symbola.ttf')

top_emoji_list= get_top_emoji(df)
pp= ''.join(top_emoji_list['emj_char']+' ')

wc= WordCloud(font_path=font_path, regexp=regexp).generate(pp)
#wcf = wc.to_file('wordcloud.png')
#image_filename = 'wordcloud.png' # replace with your own image
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

###

##combined wordcloud
total_words= person1_messages+ person2_messages
total_wc= WordCloud(font_path=font_path, regexp=regexp).generate(total_words)
#

person1_X=deque(maxlen=99999)
person1_X.append(1)
person11_X=df[1:1]
person11_X_radar=df[1:1]
i_radar_person1=0


person1_Y=deque(maxlen= 99999)
person1_Y.append(1)
person2_X=deque(maxlen=99999)
person2_X.append(1)
person2_Y=deque(maxlen= 99999)
person2_Y.append(1)
person22_X=df[1:1]
i=0
ii=0
app=dash.Dash(__name__)
app.layout=html.Div([
        dcc.Graph(id='live-graph',animate=True),
        dcc.Interval(id='graph-update',interval=1000,n_intervals=0), 
        dcc.Graph(id='live-graph2',animate=True), 
        dcc.Interval(id='graph-update2',interval=1000, n_intervals=0),
        dcc.Graph(
                id='static-graph3',
                figure={
                    'data':[go.Scatterpolar(
                            r = person1_radar_chart_weekly_df['messagecount'],
                            theta = person1_radar_chart_weekly_df['weekday'],
                            fill = 'toself'
                            ),
                            go.Scatterpolar(
                            r = person2_radar_chart_weekly_df['messagecount'],
                            theta = person2_radar_chart_weekly_df['weekday'],
                            fill = 'toself'
                            )
                    ]
                    }
                ),
        dcc.Graph(
                id='static-graph4',
                figure={
                    'data':[go.Scatterpolar(
                            r = person1_radar_chart_hourly_df['messagecount'],
                            theta = person1_radar_chart_hourly_df['hoours'],
                            fill = 'toself'
                            ),
                            go.Scatterpolar(
                            r = person2_radar_chart_hourly_df['messagecount'],
                            theta = person2_radar_chart_hourly_df['hoours'],
                            fill = 'toself'
                            )
                    ]
                    }
                ),
        dcc.Graph(
                id='static-graph5',
                figure={
                'data':[go.Pie(labels=[person1_name, person2_name], values=[person1_word_count, person2_word_count], textinfo='value')
                  ]
                }
                ),
        dcc.Graph(
                id='static-graph6',
                figure={
                'data':[go.Pie(labels=[person1_name, person2_name], values=[person1_message_count, person2_message_count], textinfo='value')
                        ]
                }
                ),
         html.Div(id="wxx",
                 children=[html.Img(id="image_wc")]
                ),
         html.Div(id="wyy",
                 children=[html.Img(id="total_wc")]
                )
                ]
        )
    

from io import BytesIO
@app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(Output('total_wc', 'src'), [Input('total_wc', 'id')])
def make_image(b):
    img = BytesIO()
    total_wc.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(Output('live-graph2','figure'),[Input('graph-update2','n_intervals')])
def update_graph_scatter2(n):
    global ii, range_date, person11_X, person22_X
    print(ii)
    person11_X= person11_X.append(df[ii:ii+1])
    sorted_words= top_words(person11_X, person1_name).tail(10)
    person3_graph= plotly.graph_objs.Bar(y=list(sorted_words['word']),x=list(sorted_words['count']),name='Scatter', orientation='h')
    
    fig = tls.make_subplots(rows=1, cols=2, shared_xaxes=False, vertical_spacing=0.009,horizontal_spacing=0.009)
    #fig.layout= go.Layout(xaxis=dict(range=[0,max(0,100)]),)
    
    fig.update_layout(
        autosize=False,
        width=1500,
        height=300,
        yaxis=dict(
            title_text="Y-axis Title",
            titlefont=dict(size=30),
        ),
    )
    
    fig.add_trace(person3_graph,row= 1, col= 1)
#    
    person22_X= person22_X.append(df[ii:ii+1])
    sorted_words= top_words(person22_X, person2_name).tail(10)
    person3_graph= plotly.graph_objs.Bar(y=list(sorted_words['word']),x=list(sorted_words['count']),name='Scatter', orientation='h')
    
    fig.add_trace(person3_graph,row= 1, col= 2)
    
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

