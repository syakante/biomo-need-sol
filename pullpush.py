import requests
import warnings
import re
import pandas as pd
import datetime
import argparse

#pullpush result: json like so
# { "data": [ ... ], "metadata": { ... }, ... }
#Im not really sure what metadata is because currently it gives some kvs of "op_a": number and stuff
#that the documentation doesn't refer to nor explain... so I guess just drop it

#to keep from data:
#body: comment text
#id: comment id
#created_utc: date
#score, maybe?

def clean_single_comment(text):
	ret = re.sub('\s+', ' ', text)
	ret = re.sub('\*+', '', ret)
	#replace usernames
	#/u/username or u/username
	ret = re.sub(r'(/?u/\S+)', '_USERNAME_', ret)
	#replace formatted hyperlinks. reddit link format is [text](link).
	ret = re.sub('\[(.+)\]\((https?:\/\/[^\)]*)\)', '\\1 \\2', ret)
	#this removes the brackets. Won't detect incorrectly formatted 
	#i also want to remove other markdown formatting stuff like italics, bold, but see next point
	#tbh when text processing you remove punctuation anyway so idk what the point of this is really. Oh well!
	#replace unformatted hyperlinks
	ret = re.sub(r'https?://\S*', '_HYPERLINK_', ret)
	return(ret)

def get_date_range(query:str, subreddit:str, startDate:str, endDate:str):
	#it's too much a PITA to try to implement both size and date range
	#since there's various possible ways one might override the other
	#lets just say date format should be YYYY-MM-DD
	start_dt = datetime.datetime.strptime(startDate, "%Y-%m-%d").replace(tzinfo = datetime.timezone.utc)
	end_dt = datetime.datetime.strptime(endDate, "%Y-%m-%d").replace(tzinfo = datetime.timezone.utc)
	if(start_dt >= end_dt):
		print("The start date is more recent or equal to the end date. Did you mean to switch them?")
		return []
	#actually if i convert everything to a datetime the timestamp() probably isn't necessary
	#kmskms
	cur_latest = end_dt
	#lt means older than, gt mean more recent than
	ret = []
	while cur_latest > start_dt:
		print("cur latest:", cur_latest, cur_latest.timestamp())
		url = ('https://api.pullpush.io/reddit/search/comment/?q='+ query
				+ '&subreddit=' + subreddit
				+ '&size=100'
				+ '&before=' + str(int(cur_latest.timestamp()))
				+ '&after=' + str(int(start_dt.timestamp())))
		print(url)
		#this gets the 100 most recent comments within the specified date range
		#i.e. L[0] is more recent than L[99]
		#so need to reduce 'before' each time
		#though after some loops, the time range will get too small that no comments get returned
		#that or during this time frame there just weren't any comments meeting this query
		try:
			got = requests.get(url).json()['data']
		except:
			print("?")
			with open("log.txt", "w") as myfile:
				myfile.write(got)
				quit()
		if(len(got) > 0):
			ret.extend(got)
			cur_latest = datetime.datetime.fromtimestamp(ret[-1]["created_utc"], tz=datetime.timezone.utc)
		else:
			cur_latest = start_dt
		#exit loop when cur_oldest is older than or equal to startDate
		#size=100 is just the max results, if fewer than 100 results satisfy the before/after it wont return 100.
	return ret


def pullpush_wrapper(query:str, subreddit:str, size:int, endDate:str = "0d"):
	#btw default sort is by newest first
	if(size < 1):
		raise Exception("Pullpush search size must be greater than 1.")
	if(size <= 100):
		url = ('https://api.pullpush.io/reddit/search/comment/?q='+ query
				+ '&subreddit=' + subreddit
				+ '&size=' + str(size)
				+ '&before=' + endDate)
		return(requests.get(url).json()["data"])
	#else
	remaining = size
	ret = []
	latest_date = endDate
	while remaining > 0:
		print("need", remaining, "more")
		cur_size = min(remaining, 100)
		url = ('https://api.pullpush.io/reddit/search/comment/?q='+ query
				+ '&subreddit=' + subreddit
				+ '&size=' + str(cur_size)
				+ '&before=' + latest_date)
		ret.extend(requests.get(url).json()["data"])
		latest_date = str(int(ret[-1]["created_utc"]))
		#im not sure date is inclusive or not and if it will result in dupe posts
		#but it doesnt look like it
		remaining -= 100	
	return ret

def f():
	#out = pullpush_wrapper(query = "helped", subreddit = "REDDITORSINRECOVERY", size = 600)
	out = get_date_range(query = "help", subreddit = "REDDITORSINRECOVERY", startDate = "2023-01-01", endDate = "2024-01-01")
	df = pd.DataFrame(out)
	df = df[['id', 'body', 'created_utc']]
	df['body'] = [clean_single_comment(x) for x in df['body']]
	df.to_csv('comments_2324.csv', encoding='utf-8-sig', index=False)

	print("wrote")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Get PullPush Reddit API results with a query and either desired number of results or date range. (both currently not supported)")
	
	parser.add_argument('query', type=str, help="Search query for comments.")
	parser.add_argument('-r', '--sub', type=str, default="REDDITORSINRECOVERY", help="Subreddit to query for comments.")
	parser.add_argument('-n', '--number', type=int, help="Number of results to query.")
	parser.add_argument('-s', '--start-date', help="Start date for date range. Should be earlier than end date.")
	parser.add_argument('-e', '--end-date', help="End date for the results. Should be more recent than start date.")
	#? for some reason string literal w/ quotes isn't working with some combination of parameters
	args = parser.parse_args()
	argnum = (args.number is not None)
	argsd = (args.start_date is not None)
	arged = (args.end_date is not None)
	if not argnum and sum([argsd, arged]) < 2:
		print("At least either number or date range is required.")
		exit(1)
	if sum([argsd, arged]) == 1:
		print("Enter both a start and end date.")
		exit(1)
	if argnum and argsd and arged:
		print("You've entered both a number and date range. Both at the same time isn't supported at this time, so only date range will be used.")
		#i think it's totally implementable using some condition like get the first n most recent results but I don't feel like it rn
	
	if argsd and arged:
		out = get_date_range(args.query, args.sub, args.start_date, args.end_date)
		
	else:
		out = pullpush_wrapper(args.query, args.sub, args.number)
	df = pd.DataFrame(out)
	df = df[['id', 'body', 'created_utc']]
	df['body'] = [clean_single_comment(x) for x in df['body']]
	fname = args.sub + '_' + '-'.join(re.findall("[a-zA-Z]+", args.query)) + '_' + args.start_date + '_' + args.end_date + '.csv'
	df.to_csv(fname, encoding='utf-8-sig', index=False)

	print("Wrote to", fname)