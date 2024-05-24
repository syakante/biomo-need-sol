import re
import pandas as pd
from html import unescape

def myfilter(s:str):
	#general
	genmatch = "(help|recommend|try|check out|look into|advise)"
	negmatch = re.compile(r"(?<=not\W|n't\W)" + genmatch, flags = re.I | re.M)
	posmatch = re.compile(r"(?<!not\W|n't\W)" + genmatch, flags = re.I | re.M)
	advise = (bool(posmatch.search(s)) or (negmatch.search(s) is not None and not (bool(negmatch.search(s)))))

	#first person
	firstPerson = re.compile(r"\b(I'?|me|my|mine)\b", flags = re.I | re.M)
	#second person imperative, I guess
	secondPerson = re.compile(r"\byou\b (should|could|can|gotta|got to|need|must)", flags = re.I | re.M)
	return bool(firstPerson.search(s)) or bool(secondPerson.search(s)) or advise

def f(infile):
	df = pd.read_csv(infile, encoding='utf-8-sig')
	#mb sth like "advice" "advise"
	#also consider if the comment anti-reccs one thing but reccs another i.e. re match logic...
	#i.e. posmatch OR not negmatch
	mask = [myfilter(x) for x in df['body']]
	df = df[mask]
	df['body'] = [unescape(x) for x in df['body']]

	outfile = infile.split('.csv')[0] + '_regexed.csv'
	df.to_csv(outfile, index=False, encoding='utf-8-sig')

f('data/REDDITORSINRECOVERY_suggest_2023-01-01_2024-01-01.csv')
print("ok...")

# negmatch = re.compile(r"(?<=not\W|n't\W)(recommend|suggest|try|check out|look into|advise)", re.MULTILINE)
# posmatch = re.compile(r"(?<!not\W|n't\W)(recommend|suggest|try|check out|look into|advise)", re.MULTILINE)
# s1 = "I wouldn't recommend xyz."
# s2 = "I recommend xyz."
# s3 = "I wouldn't recommend xyz, but I do recommend abc."
# s4 = "I recommend xyz, but I wouldn't recommend abc."
# L = [s1, s2, s3, s4]
# mask = [(bool(posmatch.search(x)) or not (bool(negmatch.search(x)))) for x in L]
# print(L)
# print(mask)