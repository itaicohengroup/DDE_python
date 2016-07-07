# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:36:31 2016

@author: Lena
"""

from sys import stdout
from time import sleep
for i in range(1,20):
    stdout.write("\r%d" % i)
    stdout.flush()
    sleep(1)
stdout.write("\n")