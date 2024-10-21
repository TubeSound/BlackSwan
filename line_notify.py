# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:35:10 2024

@author: docs9


"""

import requests
from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")




class LineNotify(object):
    
    API_URL = 'https://notify-api.line.me/api/notify'
    token ='pyBbJHApy1WeeZd67bHr0F983nnRIZzvfNJ82ek7s19'
    black_swan = ''
    
    def __init__(self):
        self.__headers = {'Authorization': 'Bearer ' + self.black_swan}

    def send(self, message, image=None, sticker_package_id=None, sticker_id=None):
        t = datetime.now().astimezone(JST)
        tstr = t.strftime('%H:%M:%S')
        payload = {
            'message': tstr + "\r " + message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
        }
        files = {}
        if image != None:
            files = {'imageFile': open(image, 'rb')}
            
        r = requests.post(
            self.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
        )
        
def test():       
    notify = LineNotify()
    notify.send('テスト')
    
if __name__ == '__main__':
    test()