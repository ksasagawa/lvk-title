"""import csv
from lxml import etree
from bs4 import BeautifulSoup as bs
import urllib
import requests 

url="https://www.youtube.com/@LukeVonKarma/videos"
html = requests.get(url)
print(html.text)
soup = bs(html.text, "lxml")

video_titles =[]

print("Cashing Video Titles ...")
entry = soup.find("yt-formatted-string")
print(entry)

print("Cashing Video Titles Done!")"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.youtube.com/@LukeVonKarma/videos")


WAIT_IN_SECONDS = 5
last_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    # Scroll to the bottom of page
    driver.execute_script("window.scrollTo(0, arguments[0]);", last_height)
    # Wait for new videos to show up
    time.sleep(WAIT_IN_SECONDS)
    
    # Calculate new document height and compare it with last height
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

titles = driver.find_elements(By.ID, "video-title")


title_list = []
for title in titles:
    title_list.append(title.text)

print(title_list)

driver.quit()

with open("titles.csv", 'w') as f:
    for t in title_list:
        try:
            f.write(t+";")
        except:
            continue
f.close()