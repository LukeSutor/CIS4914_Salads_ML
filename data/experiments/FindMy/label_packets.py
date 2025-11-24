import pyshark
import json

# CONFIG - has to be set per experiment
PCAP_FILE = 'active_queries.pcapng'  # input for this experiment
INTERVAL_SEC = 30                # location check interval in seconds 
MARGIN_SEC = 5                # margin of error in seconds 

# Prepare output
output_json = PCAP_FILE.replace('.pcapng', '.json')

# Load packets
cap = pyshark.FileCapture(PCAP_FILE, keep_packets=False)

# Find sniff timestamp of first packet
start_time = None
location_check_indices = []

for idx, pkt in enumerate(cap):
    pkt_time = float(pkt.sniff_timestamp)
    if start_time is None:
        start_time = pkt_time
    elapsed = pkt_time - start_time
    interval_num = int(elapsed // INTERVAL_SEC)
    interval_mark = start_time + interval_num * INTERVAL_SEC
    if abs(pkt_time - interval_mark) <= MARGIN_SEC:
        location_check_indices.append(idx)
    # Progress printer in case processing takes too long, can be adjusted
    if idx % 100000 == 0:
        print(f"Processed {idx} packets...")

# Save indices to JSON file
with open(output_json, 'w') as f:
    json.dump(location_check_indices, f, indent=2)

print(f"Done. Location check packet indices saved to {output_json}")
