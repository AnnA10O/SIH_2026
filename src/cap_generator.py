import uuid
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from xml.dom import minidom

# A 20km radius circle provides a v1 representative fallback for the
# localized storm footprint before we integrate the true SpatialFusionGrid polygons.
ALERT_REPRESENTATIVE_RADIUS_KM = 20.0
CAP_NAMESPACE = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER_ID = "sih.test.system@example.com"

# Register namespace so ET doesn't prepend 'ns0:' to everything
ET.register_namespace('', CAP_NAMESPACE)

def get_iso_time(dt: datetime) -> str:
    """Format datetime to ISO 8601 string for CAP (e.g., 2026-09-15T00:00:00+00:00)."""
    # Assuming UTC for simulator standard
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

def generate_sachet_cap_alert(
    station_id: str, 
    lat: float, 
    lon: float, 
    p_cb: float, 
    tier: str, 
    current_time: datetime,
    lead_time_hrs: int = 4,
    status: str = "Test"
) -> str:
    """
    Generates a valid CAP v1.2 XML string mapped to the SIH NDMA/C-DOT SACHET profile.
    """
    if tier not in ["CLOUDBURST_LIKELY", "HIGH_RISK"]:
        raise ValueError(f"Cannot generate CAP alert for non-critical tier: {tier}")

    # Tier mappings
    if tier == "CLOUDBURST_LIKELY":
        severity = "Extreme"
        urgency = "Immediate"
    else:  # HIGH_RISK
        severity = "Severe"
        urgency = "Expected"
        
    certainty = "Likely"  # Hardcoded: NEVER 'Observed' for a predictive nowcast.

    # Root Alert Element
    alert = ET.Element(f"{{{CAP_NAMESPACE}}}alert")
    
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}identifier").text = f"urn:uuid:{uuid.uuid4()}"
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}sender").text = SENDER_ID
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}sent").text = get_iso_time(current_time)
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}status").text = status
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}msgType").text = "Alert"
    ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}scope").text = "Public"
    
    # Info Element
    info = ET.SubElement(alert, f"{{{CAP_NAMESPACE}}}info")
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}category").text = "Met"
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}event").text = "Cloudburst"
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}responseType").text = "Evacuate"
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}urgency").text = urgency
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}severity").text = severity
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}certainty").text = certainty
    
    # Timings
    expires_time = current_time + timedelta(hours=lead_time_hrs)
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}effective").text = get_iso_time(current_time)
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}expires").text = get_iso_time(expires_time)
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}senderName").text = "SIH Neural Nowcaster Test System"
    
    # Human Readable Text
    headline = "High Risk of Cloudburst / Severe Flash Flooding" if tier == "HIGH_RISK" else "Extreme Cloudburst Warning / Immediate Flash Flooding"
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}headline").text = headline
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}description").text = f"Neural Nowcaster predicts a {p_cb*100:.1f}% likelihood of a cloudburst event originating near station {station_id}."
    ET.SubElement(info, f"{{{CAP_NAMESPACE}}}instruction").text = "Move to higher ground immediately. Avoid riverbanks, steep slopes, and low-lying areas. Do not attempt to cross flowing streams."
    
    # Technical Parameters
    param = ET.SubElement(info, f"{{{CAP_NAMESPACE}}}parameter")
    ET.SubElement(param, f"{{{CAP_NAMESPACE}}}valueName").text = "P_CB"
    ET.SubElement(param, f"{{{CAP_NAMESPACE}}}value").text = f"{p_cb:.3f}"
    
    # Geographic Area
    area = ET.SubElement(info, f"{{{CAP_NAMESPACE}}}area")
    ET.SubElement(area, f"{{{CAP_NAMESPACE}}}areaDesc").text = f"Region around {station_id}"
    
    # Circle format: "lat,lon radius_km"
    ET.SubElement(area, f"{{{CAP_NAMESPACE}}}circle").text = f"{lat:.4f},{lon:.4f} {ALERT_REPRESENTATIVE_RADIUS_KM}"

    # Generate pretty XML
    rough_string = ET.tostring(alert, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    # Remove empty lines introduced by minidom's toprettyxml
    pretty_xml = '\n'.join([line for line in reparsed.toprettyxml(indent='  ').split('\n') if line.strip()])
    return pretty_xml
