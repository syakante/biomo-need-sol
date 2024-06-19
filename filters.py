import re
import pandas as pd
from html import unescape

def first_filter(s:str):
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

def sec_filter(s:str):
	with open('mywordlist.txt', 'r') as f:
		mywordlist = f.readlines()
	myfilter = '|'.join(mywordlist) + ')'
	mymatch = re.compile(r"("+myfilter, flags = re.I | re.M)
	return(bool(mymatch.search(s)))

def main(filterfunc, infile, outfile=''):
	df = pd.read_csv(infile, encoding='utf-8-sig')
	#mb sth like "advice" "advise"
	#also consider if the comment anti-reccs one thing but reccs another i.e. re match logic...
	#i.e. posmatch OR not negmatch
	mask = [filterfunc(x) for x in df['body']]
	df = df[mask]
	df['body'] = [unescape(x) for x in df['body']]

	if len(outfile) < 1:
		outfile = infile.split('.csv')[0] + '_regexed.csv'
	df.to_csv(outfile, index=False, encoding='utf-8-sig')

#main(first_filter, 'data/REDDITORSINRECOVERY_suggest_2023-01-01_2024-01-01.csv')
main(sec_filter, 'data/mydocs.csv', 'second_filter.csv')
print("Done.")