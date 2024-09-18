import os
import requests
import bs4

# URL of the webpage
url = "https://www.cis.upenn.edu/~jshi/ped_html/pageshow1.html"

# Send a request to the webpage
response = requests.get(url)
soup = bs4.BeautifulSoup(response.text, 'html.parser')

# Find all image tags
img_tags = soup.find_all('img')

# Create a directory to save images
os.makedirs('images', exist_ok=True)

# Download each image
for img in img_tags:
    img_url = img['src']
    # If the image URL is relative, make it absolute
    if not img_url.startswith('http'):
        img_url = url + img_url
    # Get the image content
    img_response = requests.get(img_url)
    # Extract the image file name
    img_name = os.path.basename(img_url)
    # Save the image
    with open(os.path.join('images', img_name), 'wb') as f:
        f.write(img_response.content)
        print(img_name)

print("Images downloaded successfully.")