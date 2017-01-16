from bs4 import BeautifulSoup
import requests
import csv

ticker = raw_input("Ticker > ")

response = requests.get("https://finance.yahoo.com/quote/{}/history?p={}".format(ticker, ticker))
soup = BeautifulSoup(response.text, 'html.parser')

for data in soup.find_all("td"):
    print data["class"]

