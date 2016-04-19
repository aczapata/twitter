import psycopg2
import json
import datetime

conn = psycopg2.connect("dbname=twitter user=twitter password=12345")
cur = conn.cursor()
count = 0
tweet_path = './TwitterData240320161624.txt'
tweets_file = open(tweet_path, 'r')
for line in tweets_file:
	try:
	    tweet = json.loads(line)
	    SQL = "INSERT INTO collector_twitterdata ("
	    data = ( tweet['id_str'] , tweet['text'] , tweet['user']['screen_name'], str(tweet['created_at']))
	    s = "tweet_id, content, tweet_user, date"
	    v = ") VALUES (%s, %s, %s, %s"
	    if tweet.get('coordinates'):
	    	s += ", latitude, longitude"
	    	v += ", %s, %s"
	    	data+= (tweet['coordinates']['coordinates'][1], tweet['coordinates']['coordinates'][0])
	    if tweet['user'].get('location'):
	    	s+= ", user_location"
	    	v += ",%s"
	    	data += (tweet['user']['location'],)
	    if tweet['lang']:
	    	s += ", lang"
	    	v += ", %s"
	    	data+=(tweet['lang'],)

	    SQL= SQL + s + v+ ")"
	    #print SQL
	    #print data
	    cur.execute(SQL, data)
	    conn.commit()
	    count = count + 1
	except:
		continue
print (str(count) + " Tweets detected in "+tweet_path )