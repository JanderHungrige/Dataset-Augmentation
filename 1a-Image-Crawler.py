#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:35:52 2020

@author: base
"""
import pandas as pd
from icrawler.builtin import BingImageCrawler
from pathlib import Path

mitFilter=True
# With Bing, the type=face is not yet implemented, however my collegue pushed a fix. You can try if they already implemented the fix
filters = dict(type='photo',
               license='commercial,modify') # either photo, face, clipart, linedrawing, animated
howmany= 20
names=pd.read_csv('./Top 1000 Actors and Actresses.csv', encoding = "ISO-8859-1")
subset=names.Name

n=0   
for keyword in subset:
    n=n+1
    print(n)
    crawler = BingImageCrawler(
        parser_threads=6,
        downloader_threads=6,
        storage={'root_dir': 'Free_com_Celeb/{}'.format(keyword)}
    )    
    if mitFilter==True:
        crawler.crawl(keyword=keyword, filters=filters,max_num=howmany, min_size=(500, 500))
    else:
        crawler.crawl(keyword=keyword, max_num=howmany, min_size=(500, 500))
