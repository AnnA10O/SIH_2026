import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import glob
import os

sys.path.insert(0, 'd:/SIH')
from src.cap_generator import generate_sachet_cap_alert, CAP_NAMESPACE

print("Testing SACHET CAP v1.2 Generator...")

# 1. Test manual generation
test_time = datetime(2026, 9, 15, 12, 0, 0)
xml_str = generate_sachet_cap_alert(
    station_id="TEST_STN_01",
    lat=25.5,
    lon=85.1,
    p_cb=0.92,
    tier="CLOUDBURST_LIKELY",
    current_time=test_time,
    lead_time_hrs=4,
    status="Test"
)

# 2. Validate XML parsing and namespace
try:
    root = ET.fromstring(xml_str)
    print("XML is well-formed.")
except ET.ParseError as e:
    print(f"XML Parsing Error: {e}")
    sys.exit(1)

# Expected root tag is <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
# ET prepends the namespace to the tag in curly braces: {urn:oasis:...}alert
expected_tag = f"{{{CAP_NAMESPACE}}}alert"
if root.tag != expected_tag:
    print(f"Namespace validation failed! Expected {expected_tag}, got {root.tag}")
    sys.exit(1)
else:
    print("Namespace validation passed.")

# 3. Check for specific tags
info = root.find(f"{{{CAP_NAMESPACE}}}info")
if info is None:
    print("Validation failed: <info> tag missing")
    sys.exit(1)

status = root.find(f"{{{CAP_NAMESPACE}}}status")
if status is None or status.text != "Test":
    print(f"Validation failed: <status> tag missing or not 'Test' (got {status.text if status is not None else 'None'})")
    sys.exit(1)
    
certainty = info.find(f"{{{CAP_NAMESPACE}}}certainty")
if certainty is None or certainty.text != "Likely":
    print(f"Validation failed: <certainty> tag missing or not 'Likely' (got {certainty.text if certainty is not None else 'None'})")
    sys.exit(1)

sender = root.find(f"{{{CAP_NAMESPACE}}}sender")
if sender is None or "example.com" not in sender.text:
    print(f"Validation failed: <sender> does not contain example.com (got {sender.text if sender is not None else 'None'})")
    sys.exit(1)

print("All validations passed. XML snippet:")
print("\n".join(xml_str.split("\n")[:15]))
print("...")
