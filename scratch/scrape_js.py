import requests
import re
import urllib3
urllib3.disable_warnings()
url = 'https://mosdac.gov.in/main-FNVDTK32.js'
try:
    r = requests.get(url, verify=False, timeout=10)
    print(r.status_code, len(r.text))
    aws = re.findall(r'[\"\']([A-Z0-9_]*AWS[A-Z0-9_]*)[\"\']', r.text)
    if aws: print('AWS datasets:', set(aws))
    datasetIds = re.findall(r'datasetId[\"\'\s:=]*([A-Z0-9_]+)', r.text)
    if datasetIds: print('datasetIds:', set(datasetIds))
except Exception as e:
    print('Error:', e)
