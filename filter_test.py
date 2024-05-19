import re
import pandas as pd

def f(infile):
	df = pd.read_csv(infile, encoding='utf-8-sig')
	negmatch = re.compile(r"(?<!not\W|n't\W)(recommend|suggest|try|check out|look into|advise)", re.MULTILINE)
	posmatch = re.compile(r"(recommend|suggest|try|check out|look into|advise)", re.MULTILINE)
	#mb sth like "advice" "advise"
	#also consider if the comment anti-reccs one thing but reccs another i.e. re match logic...

	mask = [myfilter.search(x) is not None for x in df['body']]
	df = df[mask]
	outfile = infile.split('.csv')[0] + '_regexed.csv'
	df.to_csv(outfile, index=False, encoding='utf-8-sig')