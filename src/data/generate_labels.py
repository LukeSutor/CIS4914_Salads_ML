#!/usr/bin/env python3
"""
Script to automatically generate location data labels for PCAP files.

This script analyzes PCAP files in the train and val folders and creates JSON labels
indicating which packets contain location sharing requests. It identifies location
sharing based on:
1. Hostnames in locationsharing_hosts.txt
2. Find My Friends servers matching pattern: ^(p[0-9]{1,3}-fmfmobile.icloud.com|p[0-9]{1,3}-fmf.icloud.com)$

The JSON output contains 1-indexed packet numbers for compatibility with Wireshark.

Usage:
    python generate_labels.py [--config CONFIG_PATH] [--force] [--folder FOLDER_PATH]
    
Examples:
    # Generate labels for all folders in config
    python generate_labels.py
    
    # Generate labels for a specific folder
    python generate_labels.py --folder "C:\\path\\to\\data\\train"
    
    # Force overwrite existing labels
    python generate_labels.py --force
"""

from __future__ import annotations

import functools
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import dpkt
from tqdm import tqdm

# Add the src directory to the path to enable absolute imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.config import load_config
from data.pcap_utils import _read_pcap


def load_location_hosts(hosts_file: Path) -> Set[str]:
    """Load location sharing host patterns from file."""
    hosts = set()
    if hosts_file.exists():
        with hosts_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    hosts.add(line.lower())
    return hosts


def load_ip_cache(cache_file: Path) -> Dict[str, List[str]]:
    """Load cached hostname-to-IPs mapping from file.
    
    Returns:
        Dictionary mapping hostnames to lists of IP addresses
    """
    if not cache_file.exists():
        return {}
    
    try:
        with cache_file.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load IP cache: {e}")
        return {}


def save_ip_cache(cache_file: Path, ip_cache: Dict[str, List[str]]) -> None:
    """Save hostname-to-IPs mapping to cache file."""
    try:
        with cache_file.open('w', encoding='utf-8') as f:
            json.dump(ip_cache, f, indent=2)
        print(f"Saved IP cache to {cache_file.name}")
    except IOError as e:
        print(f"Warning: Could not save IP cache: {e}")


def build_ip_lookup_from_hosts(
    location_hosts: Set[str],
    cache_file: Path,
    force_refresh: bool = False
) -> Dict[str, str]:
    """Build IP-to-hostname lookup table, using cache when available.
    
    Args:
        location_hosts: Set of hostnames to resolve
        cache_file: Path to cache file for storing resolutions
        force_refresh: If True, ignore cache and re-resolve all hosts
        
    Returns:
        Dictionary mapping IP addresses to hostnames (for fast lookup)
    """
    # Load existing cache
    hostname_to_ips = load_ip_cache(cache_file) if not force_refresh else {}
    
    # Determine which hosts need resolution
    cached_hosts = set(hostname_to_ips.keys())
    hosts_to_resolve = location_hosts - cached_hosts
    
    if hosts_to_resolve:
        print(f"\nResolving {len(hosts_to_resolve)} new/uncached hosts...")
        print(f"(Using {len(cached_hosts)} cached hosts)")
        
        resolved_count = 0
        failed_count = 0
        
        with tqdm(list(hosts_to_resolve), desc="  Resolving hosts", unit="host") as pbar:
            for hostname in pbar:
                try:
                    # Skip entries that are already IP addresses
                    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', hostname):
                        hostname_to_ips[hostname] = [hostname]
                        resolved_count += 1
                        pbar.set_postfix(resolved=resolved_count, failed=failed_count)
                        continue
                    
                    # Resolve hostname to IP(s)
                    socket.setdefaulttimeout(2.0)
                    ips = socket.gethostbyname_ex(hostname)[2]
                    hostname_to_ips[hostname] = ips
                    resolved_count += 1
                    pbar.set_postfix(resolved=resolved_count, failed=failed_count)
                    
                except (socket.herror, socket.gaierror, OSError, socket.timeout):
                    # Store empty list for hosts that don't resolve
                    hostname_to_ips[hostname] = []
                    failed_count += 1
                    pbar.set_postfix(resolved=resolved_count, failed=failed_count)
                finally:
                    socket.setdefaulttimeout(None)
        
        print(f"  Successfully resolved: {resolved_count}")
        print(f"  Failed to resolve: {failed_count}")
        
        # Save updated cache
        save_ip_cache(cache_file, hostname_to_ips)
    else:
        print(f"\nUsing cached IP resolutions for all {len(cached_hosts)} hosts")
    
    # Build reverse lookup table: IP -> hostname
    ip_to_hostname = {}
    total_ips = 0
    for hostname, ips in hostname_to_ips.items():
        for ip in ips:
            ip_to_hostname[ip] = hostname
            total_ips += 1
    
    print(f"Built lookup table with {total_ips} IP addresses from {len(hostname_to_ips)} hosts\n")
    return ip_to_hostname


def is_find_my_friends_host(hostname: str) -> bool:
    """Check if hostname matches Find My Friends server pattern."""
    pattern = r'^(p[0-9]{1,3}-fmfmobile\.icloud\.com|p[0-9]{1,3}-fmf\.icloud\.com)$'
    return bool(re.match(pattern, hostname.lower()))


# Global cache for DNS lookups to avoid repeated expensive operations
_dns_cache: Dict[str, str] = {}

def resolve_ip_to_hostname(ip: str) -> str:
    """Attempt to resolve IP address to hostname with caching."""
    # Check cache first
    if ip in _dns_cache:
        return _dns_cache[ip]
    
    try:
        # Set a timeout to avoid hanging on slow DNS servers
        socket.setdefaulttimeout(2.0)  # 2 second timeout
        hostname = socket.gethostbyaddr(ip)[0]
        result = hostname.lower()
    except (socket.herror, socket.gaierror, OSError, socket.timeout):
        result = ""
    finally:
        socket.setdefaulttimeout(None)  # Reset to default
    
    # Cache the result (including empty results to avoid retry)
    _dns_cache[ip] = result
    return result


def analyze_pcap_for_location_requests(
    pcap_path: Path, 
    location_hosts: Set[str],
    ip_lookup: Dict[str, str]
) -> List[int]:
    """
    Analyze PCAP file and return 1-indexed packet numbers containing location requests.
    
    Args:
        pcap_path: Path to PCAP file
        location_hosts: Set of known location sharing hostnames
        ip_lookup: Pre-resolved IP to hostname mapping
        
    Returns:
        List of 1-indexed packet numbers with location sharing activity
    """
    location_packets = []
    packet_num = 0
    
    print(f"  Analyzing {pcap_path.name}...")
    
    # Detect link layer type from first packet
    link_type = None
    
    try:
        # Read all packets into memory first to enable progress tracking
        packets = list(_read_pcap(pcap_path))
        total_packets = len(packets)
        
        # Detect link layer type from the PCAP file
        with pcap_path.open("rb") as f:
            try:
                pcap_reader = dpkt.pcap.Reader(f)
                link_type = pcap_reader.datalink()
            except (ValueError, OSError):
                # Try pcapng
                f.seek(0)
                try:
                    pcapng_reader = dpkt.pcapng.Reader(f)
                    link_type = getattr(pcapng_reader, 'datalink', lambda: 1)()
                except:
                    link_type = 1  # Default to Ethernet
        
        print(f"    Detected link layer type: {link_type}")
        
        # Process packets with progress bar
        with tqdm(packets, desc=f"    Processing {pcap_path.name}", 
                 unit="pkt", leave=False, total=total_packets) as pbar:
            for ts, buf in pbar:
                packet_num += 1
                
                try:
                    # Parse packet based on link layer type
                    ip = None
                    
                    # DLT_EN10MB (1) = Ethernet
                    # DLT_RAW (101, 12, 14) = Raw IP (no link layer)
                    # DLT_LINUX_SLL (113) = Linux cooked capture
                    # DLT_IPV4 (228) = Raw IPv4
                    # DLT_IPV6 (229) = Raw IPv6
                    
                    if link_type == 1:  # Ethernet
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                            ip = eth.data
                    elif link_type in (12, 14, 101):  # Raw IP
                        # Try IPv4 first
                        try:
                            ip = dpkt.ip.IP(buf)
                        except:
                            # Try IPv6
                            try:
                                ip = dpkt.ip6.IP6(buf)
                            except:
                                continue
                    elif link_type == 113:  # Linux cooked capture
                        # Linux cooked capture has 16 byte header
                        if len(buf) < 16:
                            continue
                        # Protocol type is at offset 14-15
                        proto_type = int.from_bytes(buf[14:16], 'big')
                        payload = buf[16:]
                        if proto_type == 0x0800:  # IPv4
                            try:
                                ip = dpkt.ip.IP(payload)
                            except:
                                continue
                        elif proto_type == 0x86DD:  # IPv6
                            try:
                                ip = dpkt.ip6.IP6(payload)
                            except:
                                continue
                    elif link_type == 228:  # Raw IPv4
                        try:
                            ip = dpkt.ip.IP(buf)
                        except:
                            continue
                    elif link_type == 229:  # Raw IPv6
                        try:
                            ip = dpkt.ip6.IP6(buf)
                        except:
                            continue
                    else:
                        # Try Ethernet as default fallback
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                                ip = eth.data
                        except:
                            continue
                    
                    # Skip if we couldn't parse an IP packet
                    if ip is None or not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                        continue
                    
                    # Get source and destination IPs
                    try:
                        if isinstance(ip, dpkt.ip.IP):
                            src_ip = socket.inet_ntoa(ip.src)
                            dst_ip = socket.inet_ntoa(ip.dst)
                        else:  # IPv6
                            src_ip = socket.inet_ntop(socket.AF_INET6, ip.src)
                            dst_ip = socket.inet_ntop(socket.AF_INET6, ip.dst)
                    except (OSError, ValueError) as e:
                        # Skip packets with malformed IP addresses
                        continue
                
                    import pdb; pdb.set_trace()
                    
                    # Check if this packet contains location sharing traffic
                    is_location_packet = False
                    
                    # Method 1: Fast IP lookup in pre-resolved table
                    for check_ip in [src_ip, dst_ip]:
                        if check_ip in ip_lookup:
                            is_location_packet = True
                            break
                    
                    # Method 2: Check direct IP matches against location hosts
                    if not is_location_packet:
                        for check_ip in [src_ip, dst_ip]:
                            if check_ip in location_hosts:
                                is_location_packet = True
                                break
                    
                    # Method 3: Extract hostname from packet content and check patterns
                    if not is_location_packet and hasattr(ip, 'data'):
                        hostnames = extract_hostnames_from_packet(ip)
                        for hostname in hostnames:
                            if hostname in location_hosts or is_find_my_friends_host(hostname):
                                is_location_packet = True
                                break
                    
                    # Method 4: Try reverse DNS lookup on destination IPs (expensive, last resort)
                    if not is_location_packet:
                        for check_ip in [dst_ip]:  # Only check destination to reduce lookups
                            hostname = resolve_ip_to_hostname(check_ip)
                            if hostname and (hostname in location_hosts or is_find_my_friends_host(hostname)):
                                is_location_packet = True
                                break
                    
                    if is_location_packet:
                        location_packets.append(packet_num)
                        # Update progress bar description to show found packets
                        pbar.set_postfix(found=len(location_packets))
                        
                except Exception as e:
                    # Skip malformed packets
                    continue
                finally:
                    pbar.update(1)
                    
    except Exception as e:
        print(f"    Error reading {pcap_path}: {e}")
        return []
    
    print(f"    Found {len(location_packets)} location sharing packets out of {packet_num} total packets")
    return location_packets


def extract_hostnames_from_packet(ip_packet) -> List[str]:
    """Extract all possible hostnames from an IP packet."""
    hostnames = []
    
    try:
        if not hasattr(ip_packet, 'data'):
            return hostnames
            
        transport = ip_packet.data
        
        # Handle TCP and UDP
        if isinstance(transport, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            if hasattr(transport, 'data') and transport.data:
                packet_data = transport.data
                
                # Extract from HTTP headers
                try:
                    packet_str = packet_data.decode('utf-8', errors='ignore')
                    
                    # HTTP Host header
                    host_matches = re.findall(r'host:\s*([^\r\n\s]+)', packet_str, re.IGNORECASE)
                    hostnames.extend([h.strip().lower() for h in host_matches])
                    
                    # Extract domain-like patterns (for TLS SNI and other protocols)
                    domain_pattern = r'([a-z0-9.-]+\.(?:com|net|org|edu|gov|mil|int|co|uk|de|fr|jp|cn|ru|io|app|cloud|icloud))\b'
                    domain_matches = re.findall(domain_pattern, packet_str, re.IGNORECASE)
                    hostnames.extend([d.strip().lower() for d in domain_matches])
                    
                except (UnicodeDecodeError, AttributeError):
                    pass
                
                # Try to extract TLS SNI (Server Name Indication) more specifically
                hostnames.extend(extract_tls_sni(packet_data))
                
    except Exception:
        pass
    
    # Clean and deduplicate hostnames
    clean_hostnames = []
    for hostname in hostnames:
        hostname = hostname.strip().lower()
        if hostname and '.' in hostname and len(hostname) > 3:
            # Basic validation - must be a reasonable hostname
            if re.match(r'^[a-z0-9.-]+$', hostname) and not hostname.startswith('.') and not hostname.endswith('.'):
                clean_hostnames.append(hostname)
    
    return list(set(clean_hostnames))  # Remove duplicates


def extract_tls_sni(packet_data: bytes) -> List[str]:
    """Extract Server Name Indication from TLS packets."""
    hostnames = []
    
    try:
        # Look for TLS handshake patterns
        # This is a simplified approach - full TLS parsing would be more complex
        data = packet_data
        
        # Look for TLS handshake (0x16) followed by version
        if len(data) < 50:  # Too short for TLS handshake
            return hostnames
            
        # Search for SNI extension (type 0x0000)
        i = 0
        while i < len(data) - 10:
            # Look for SNI extension pattern
            if (data[i:i+2] == b'\x00\x00' and  # SNI extension type
                i + 9 < len(data)):
                
                # Try to read the hostname length and data
                try:
                    # Skip extension length fields
                    hostname_len_offset = i + 7
                    if hostname_len_offset + 2 < len(data):
                        hostname_len = int.from_bytes(data[hostname_len_offset:hostname_len_offset+2], 'big')
                        if (hostname_len > 0 and hostname_len < 253 and 
                            hostname_len_offset + 2 + hostname_len <= len(data)):
                            hostname_bytes = data[hostname_len_offset+2:hostname_len_offset+2+hostname_len]
                            hostname = hostname_bytes.decode('utf-8', errors='ignore').lower().strip()
                            if hostname and '.' in hostname:
                                hostnames.append(hostname)
                except Exception:
                    pass
            i += 1
            
    except Exception:
        pass
    
    return hostnames


def process_folder(
    folder_path: Path, 
    location_hosts: Set[str],
    ip_lookup: Dict[str, str],
    force_overwrite: bool = False
) -> Tuple[int, int]:
    """Process all PCAP files in a folder and generate labels.
    
    Returns:
        Tuple of (total_processed, total_labeled)
    """
    print(f"\nProcessing folder: {folder_path}")
    
    if not folder_path.exists():
        print(f"  Folder does not exist: {folder_path}")
        return 0, 0
    
    # Find all PCAP files
    pcap_files = list(folder_path.glob("*.pcap")) + list(folder_path.glob("*.pcapng"))
    
    if not pcap_files:
        print(f"  No PCAP files found in {folder_path}")
        return 0, 0
    
    print(f"  Found {len(pcap_files)} PCAP files")
    
    processed = 0
    labeled = 0
    
    for pcap_file in pcap_files:
        processed += 1
        
        # Determine corresponding JSON file path
        json_file = pcap_file.with_suffix('.json')
        
        # Skip if JSON already exists and we're not forcing overwrite
        if json_file.exists() and not force_overwrite:
            print(f"  Skipping {pcap_file.name} - labels already exist")
            continue
        
        # Analyze PCAP file
        location_packets = analyze_pcap_for_location_requests(pcap_file, location_hosts, ip_lookup)
        
        # Write labels to JSON file
        try:
            with json_file.open('w', encoding='utf-8') as f:
                json.dump(location_packets, f, indent=2)
            print(f"  Created labels: {json_file.name}")
            labeled += 1
        except Exception as e:
            print(f"  Error writing {json_file}: {e}")
    
    return processed, labeled


def main():
    """Main function to generate labels for all PCAP files."""
    import argparse
    
    # Get absolute paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    default_config = project_root / "configs" / "training.yaml"
    
    parser = argparse.ArgumentParser(description="Generate location sharing labels for PCAP files")
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to training configuration file")
    parser.add_argument("--force", action="store_true", 
                       help="Overwrite existing label files")
    parser.add_argument("--folder", type=str, 
                       help="Process only a specific folder (overrides config)")
    parser.add_argument("--refresh-cache", action="store_true",
                       help="Force refresh the IP resolution cache (re-resolve all hosts)")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Load location sharing hosts
    script_dir = Path(__file__).parent
    hosts_file = script_dir / "locationsharing_hosts.txt"
    location_hosts = load_location_hosts(hosts_file)
    print(f"Loaded {len(location_hosts)} known location sharing hosts")
    
    # Build IP lookup table from cached resolutions
    cache_file = script_dir / "locationsharing_ips_cache.json"
    ip_lookup = build_ip_lookup_from_hosts(location_hosts, cache_file, force_refresh=args.refresh_cache)
    
    # Get data folders
    if args.folder:
        all_folders = [args.folder]
    else:
        train_folders = config.get('data', {}).get('train_folders', [])
        val_folders = config.get('data', {}).get('val_folders', [])
        all_folders = train_folders + val_folders
    
    if not all_folders:
        print("No data folders specified")
        return
    
    print(f"Processing {len(all_folders)} folders...")
    
    # Process each folder
    total_processed = 0
    total_labeled = 0
    
    for folder_str in all_folders:
        folder_path = Path(folder_str)
        processed, labeled = process_folder(folder_path, location_hosts, ip_lookup, force_overwrite=args.force)
        total_processed += processed
        total_labeled += labeled
    
    print(f"\nLabel generation complete!")
    print(f"Processed {total_processed} PCAP files")
    print(f"Generated labels for {total_labeled} files")


if __name__ == "__main__":
    main()