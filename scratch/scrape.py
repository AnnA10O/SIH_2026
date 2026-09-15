import requests
import re
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

def scrape(url):
    try:
        r = requests.get(url, verify=False, timeout=10)
        print(f'=== {url} ({r.status_code}) ===')
        soup = BeautifulSoup(r.text, 'html.parser')
        scripts = soup.find_all('script')
        for s in scripts:
            if s.get('src'):
                print('Found script:', s['src'])
        
        matches = re.findall(r'datasetId[\"\'\s:=]*([A-Z0-9_]+)', r.text)
        if matches:
            print('datasetIds in HTML:', set(matches))
            
        aws_matches = re.findall(r'[\"\']([A-Z0-9_]*AWS[A-Z0-9_]*)[\"\']', r.text)
        if aws_matches:
            print('AWS literals in HTML:', set(aws_matches))
            
    except Exception as e:
        print('Error:', e)

scrape('https://mosdac.gov.in/internal/catalog-insitu')
scrape('https://mosdac.gov.in/insitu')
