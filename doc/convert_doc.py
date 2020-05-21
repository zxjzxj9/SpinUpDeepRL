""" Conver Latex Equations in README.md into URL
"""

import urllib
import re
import argparse

prefix = r"https://microsoft.codecogs.com/svg.latex?"
eqn = r"($$)(.+)($$)"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str, default="", help="Source file")
parser.add_argument("-d", "--dest", type=str, default="", help="Destination file")
opt = parser.parse_args()

def replace(pattern):
    a = pattern.group(0)
    t = pattern.group(2)
    u = prefix + urllib.parse.urlencode(t)
    s = "![Eqn]({:s})\n<!--{:s}-->\n".format(u, a)
    return s

def convert(src, dst):
    sstr = open(src, "r").read()
    
    