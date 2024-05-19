import praw
from praw.models import MoreComments

with open('reddit_headers.txt') as f:
	c_id = f.readline().strip()
	c_se = f.readline().strip()

leddit = praw.Reddit(
	client_id = c_id,
	client_secret = c_se,
	password = "",
	user_agent = "scrape",
	username = "")

#url = "https://reddit.com/r/REDDITORSINRECOVERY/comments/1bi6sag/who_here_hit_rock_bottom_from_drug_addiction_in/"

def single_post_comments(url):
	submission = leddit.submission(url=url)

	posts = []
	for top_level_comment in submission.comments:
		if isinstance(top_level_comment, MoreComments):
			continue
		posts.append(top_level_comment.body)
	print("Found", len(posts), "top level comments.")
	return(posts)

def sample_sub_comments():
	sub_comments = []
	c = 0
	for submission in leddit.subreddit("REDDITORSINRECOVERY").top("week"):
		#le nested for loop zany emoji
		print("Post", c)
		c += 1
		for top_level_comment in submission.comments:
			if isinstance(top_level_comment, MoreComments):
				continue
			sub_comments.append(top_level_comment.body)
		print("Found", len(sub_comments), "top level comments from the sub.")

	import random
	random.seed(0)
	mysample = random.sample(sub_comments, 10)
	with open("test_comments.txt", 'w', encoding = 'utf-8') as file:
		for s in mysample:
			file.write(s + '\n')
			#in the future choose some delimiter (like a csv instead of txt) bc comments can have newlines
	print("ok")
	return
sample_sub_comments()