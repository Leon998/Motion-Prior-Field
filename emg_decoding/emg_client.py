import redis


rds = redis.Redis(host='localhost', port=6379, decode_responses=True)

while True:
    i = rds.get('action')
    print(i)