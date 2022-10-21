from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler

from linebot.exceptions import InvalidSignatureError

from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot import LineBotApi, WebhookHandler
#from linebot.exceptions import InvalidSignatureerror
from linebot.models import *
from urllib.parse import parse_qsl
import datetime
import tensorflow as tf
import os
import csv
import math
from datetime import datetime, timezone, timedelta
from translate import Translator
from urllib.parse import quote
#import speech_recognition as sr
#import wave, pyaudio
#from gtts import gTTs
#from pygame import mixer
#from pydub import AudioSegment
#import ffmpeg
#import xml.etree.cElementtree as ET


app = Flask(__name__)

line_bot_api = LineBotApi('m8WFb89a2M99Upf794KC2dml0SxyAA5rx1vyXlT4DGyCsuGzRlhhTpZtXKzjxMGXi4Hn3BzcEaEkeuT9SLTzv2MHCbmkUh9d0hi/L4sAkjCTEe/2OruCIH/siusm/sI5ooL07x9vOCjwpIZDhPJDrgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('f51a2963d431ceb20e35c259e2178c08')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
       handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))

@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, AudioMessage ))
def handle_content_message(event):
    import cv2
    from PIL import Image
    from keras.models import Model, load_model
    import keras
    import numpy as np
    #import matplotlib.pyplot as plt
    #from keras.datasets import mnist
    from keras.models import Model, load_model, Sequential
    from keras.layers import Input, add
    from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape
    from keras import regularizers
    from keras.regularizers import l2
    from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
    from keras.utils import np_utils
    from keras import losses
    user_id = event.source.user_id
    print("user_id =" , user_id)
    tz =timezone(timedelta(hours=+8))
    now_t=str(datetime.now(tz))#datetime.now(tz).isoformat()
    t_=now_t.replace(" ","-")
    t_=t_.replace(":","-")
    print("now ymdhms:"+t_)
    
    print("now_hr:"+t_[11:13])
    print('here is handle_content_message')
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        ext = 'm4a'
    else:
        return
        
    message_content = line_bot_api.get_message_content(event.message.id)
    
    if ext == 'jpg':
        filename = str(user_id)+'_'+t_+'.jpg'
        with open(filename, 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)
            
            
            
            
    picSize1=28
    picSize2=28
    
    image = Image.open(filename)
    image = np.asarray(image)
    #sum_im-image.sum()
    #print("sum_im=",sum_im)
    image = cv2.resize(image,dsize=(picSize2,picSize1), interpolation=cv2.INTER_CUBIC)/255.0
    im = image[:,:,0].astype('float32') #轉2維
    im_ts = im.reshape(1, 28*28)#轉2維
    tmp=str(im_ts.shape)
    
    input_size=28*28 #數字的影像長寬 28*28
    
    hidden_size=32
    code_size=100
    x=Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(x)
    h=Dense(code_size,activation='relu')(hidden_1)
    h=Dropout(0.25)(h)
    #from keras import backend as K
    #h_shape=K.int_shape(h)
    #print("h_shape: ",h _shape)
    hidden_2 = Dense(hidden_size,activation='tanh')(h)
    r = Dense(1, activation='relu')(hidden_2)
    model = Model(inputs=x, outputs=r)
    model=load_model("mnist.h5")
    model.compile(optimizer='adam', loss='mse')
    y_pred = model.predict(im_ts[0:1,:])
    y_pred = str(np.squeeze(y_pred))
    #print("y_pred=",y_pred)
    y_pred = float(y_pred)
    y_pred = math.ceil(y_pred)
    line_bot_api.reply_message(event.reply_token,TextSendMessage(text=y_pred))


if __name__ == '__main__':
    app.run()