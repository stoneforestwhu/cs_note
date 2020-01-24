# -*- coding: utf-8 -*- 
import os 

with open('temp.txt', 'r', encoding='utf-8') as rdFile:
    with open('temp_new.txt', 'w', encoding='utf-8') as wdFile:
        for line in rdFile:
            line = line.strip() + ' '
            wdFile.write(line)