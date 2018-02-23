import json

with open('/Users/sean/Cambridge/yelp_dataset/photos.txt', 'r') as f:
    image_json_dict = json.load(f)

with open('data.json', 'w') as fp:
    json.dump({}, fp)