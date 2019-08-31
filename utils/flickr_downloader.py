"""Script for downloading of the dataset from flickr"""

import urllib.request
import os

import flickrapi


API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET')
N = 1500  # change keyword after N (they tend to duplicate after 1500)
KEYWORDS = ['detail', 'patterns', 'artist', 'animal']

flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, cache=True)

urls = []

for keyword in KEYWORDS:
    photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword,
                         extras='url_c', per_page=100, sort='relevance')

    i = 0

    for nr, photo in enumerate(photos):
        print(f"Searching for '{keyword}': {nr}")
        url = photo.get('url_c')
        if url and url not in urls:
            urls.append(url)
            i += 1
        if i > N:
            break

for nr, url in enumerate(urls):
    print('Downloading and saving...', nr)
    urllib.request.urlretrieve(url, f"{nr}.jpg")
