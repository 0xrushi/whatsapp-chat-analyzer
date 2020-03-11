#https://www.youtube.com/watch?v=37Zj955LFT0&list=PLQVvvaa0QuDfsGImWNt1eUEveHOepkjqt&index=4
import dash
from dash.dependencies import Output,Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls

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
person1_Y=deque(maxlen= 99999)
person1_Y.append(1)
person2_X=deque(maxlen=99999)
person2_X.append(1)
person2_Y=deque(maxlen= 99999)
person2_Y.append(1)


i=0
app=dash.Dash(__name__)
app.layout=html.Div([dcc.Graph(id='live-graph',animate=True),dcc.Interval(id='graph-update',interval=1000,n_intervals=0),])


@app.callback(Output('live-graph','figure'),[Input('graph-update','n_intervals')])
def update_graph_scatter(n):
    global i, range_date
    print(i)
    person1_X.append(person1_df['Date2'][i])
    person1_Y.append(person1_df['messagecount'][i])
    person2_X.append(person2_df['Date2'][i])
    person2_Y.append(person2_df['messagecount'][i])
    i=i+1
    #data=plotly.graph_objs.Bar(x=list(X),y=list(Y),name='Scatter')
    person1_graph= plotly.graph_objs.Scatter(x=list(person1_X), y=list(person1_Y), name="PERSON1",
                         line_color='dimgray', mode= 'lines+markers')
    
    person2_graph= plotly.graph_objs.Scatter(x=list(person2_X), y=list(person2_Y), name="PERSON2",
                         line_color='red', mode= 'lines+markers')
#    person1_graph= {'x':list(person1_X), 'y':list(person1_Y), 'type': 'scatter', 'name': 'Boats'}
#    person1_graph= {'x':["av", "bc", "cd", "ef"], 'y':[1,2,3,4], 'type': 'scatter', 'name': 'Boats'}
    fig = tls.make_subplots(rows=3, cols=1, shared_xaxes=True,vertical_spacing=0.009,horizontal_spacing=0.009)
    fig.append_trace(person1_graph,1,1)
    fig.append_trace(person2_graph,1,1)
    fig.layout= go.Layout(xaxis=dict(range=(min(pd.to_datetime(range_date)), max(pd.to_datetime(range_date)))),yaxis=dict(range=[0,max(person1_Y)]),)
    return fig
#    return{'data':[person1_graph],'layout':go.Layout(xaxis=dict(range=(min(pd.to_datetime(range_date)), max(pd.to_datetime(range_date)))),yaxis=dict(range=[0,max(person1_Y)]),)}

if __name__=='__main__':
    app.run_server(debug=True)
