""" Conver Latex Equations in README.md into URL
"""

from urllib import parse
import re
import argparse

prefix = r"https://microsoft.codecogs.com/svg.latex?"
eqn = r"(\$\$)(.+)(\$\$)"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, default="", help="Source file")
parser.add_argument("-d", "--dst", type=str, default="", help="Destination file")
opt = parser.parse_args()

def replace(pattern):
    a = pattern.group(0)
    t = pattern.group(2)
    u = prefix + parse.quote(t)
    s = "![Eqn]({:s})\n<!--{:s}-->\n".format(u, a)
    return s

def convert(src, dst):
    with open(src, "r") as s, open(dst, "w") as d:
        for line in s:
            d.write(re.sub(eqn, replace, line))

if __name__ == "__main__":
    convert(opt.src, opt.dst)
