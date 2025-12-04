from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_hash32(x: str) -> int:
    return int(hashlib.blake2b(x.encode("utf-8"), digest_size=4).hexdigest(), 16)


def _is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:
        return False


@dataclass
class PacketFeatures:
    """Holds per-packet numeric features and metadata."""

    length: int
    l4_tcp: int
    l4_udp: int
    l4_icmp: int
    direction_out: int  # 1 if device -> remote, 0 if remote -> device, -1 if unknown
    src_port: int
    dst_port: int
    tcp_syn: int
    tcp_ack: int
    tcp_fin: int
    tcp_rst: int
    flow_hash: int  # 32-bit hash of 5-tuple

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "length",
            "l4_tcp",
            "l4_udp",
            "l4_icmp",
            "direction_out",
            "src_port",
            "dst_port",
            "tcp_syn",
            "tcp_ack",
            "tcp_fin",
            "tcp_rst",
            "flow_hash",
        ]


def _read_pcap(path: Path) -> Iterable[Tuple[float, bytes]]:
    """Read packets from pcap/pcapng using dpkt when available.

    We lazily import dpkt to keep import time small when not needed.
    """
    import dpkt  # type: ignore

    with path.open("rb") as f:
        # Try pcap first, then pcapng
        try:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                yield float(ts), buf
            return
        except (ValueError, OSError):
            pass

    # Retry as pcapng
    import dpkt.pcapng  # type: ignore

    with path.open("rb") as f:
        pcapng = dpkt.pcapng.Reader(f)
        for ts, buf in pcapng:
            yield float(ts), buf


def _parse_packet(ts: float, buf: bytes) -> Tuple[PacketFeatures, Dict[str, str]]:
    """Parse a single packet into features using dpkt layers.

    Returns:
        PacketFeatures and a small metadata dict with string IPs to determine direction.
    """
    import dpkt  # type: ignore

    length = len(buf)
    l4_tcp = l4_udp = l4_icmp = 0
    src_ip = dst_ip = None
    src_port = dst_port = 0
    tcp_syn = tcp_ack = tcp_fin = tcp_rst = 0
    flow_hash = 0

    ip = None
    
    # Try Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            ip = eth.data
    except Exception:
        pass

    # Try SLL (Linux Cooked Capture) if not Ethernet IP
    if ip is None:
        try:
            sll = dpkt.sll.SLL(buf)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                ip = sll.data
        except Exception:
            pass

    # Try Loopback if still nothing
    if ip is None:
        try:
            loop = dpkt.loopback.Loopback(buf)
            if isinstance(loop.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                ip = loop.data
        except Exception:
            pass

    # Try Raw IP (heuristic)
    if ip is None:
        if len(buf) >= 1:
            v = buf[0] >> 4
            if v == 4:
                try:
                    ip = dpkt.ip.IP(buf)
                except Exception:
                    pass
            elif v == 6:
                try:
                    ip = dpkt.ip6.IP6(buf)
                except Exception:
                    pass

    if ip is not None:
        try:
            if isinstance(ip, dpkt.ip.IP):
                src_ip = ipaddress.ip_address(ip.src).compressed
                dst_ip = ipaddress.ip_address(ip.dst).compressed
            elif isinstance(ip, dpkt.ip6.IP6):
                src_ip = ipaddress.ip_address(ip.src).compressed
                dst_ip = ipaddress.ip_address(ip.dst).compressed
        except ValueError:
            pass
    else:
        # Non-IP traffic or unparseable
        pf = PacketFeatures(
            length=length,
            l4_tcp=0,
            l4_udp=0,
            l4_icmp=0,
            direction_out=-1,
            src_port=0,
            dst_port=0,
            tcp_syn=0,
            tcp_ack=0,
            tcp_fin=0,
            tcp_rst=0,
            flow_hash=0,
        )
        return pf, {"src_ip": "", "dst_ip": ""}

    # L4
    try:
        if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
             # For Raw IP, ip is the packet itself, so ip.data is the payload
             l4_payload = ip.data
        else:
             # Should not happen given logic above, but for safety
             l4_payload = ip.data

        if isinstance(l4_payload, dpkt.tcp.TCP):
            l4_tcp = 1
            tcp = l4_payload
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            tcp_syn = int((tcp.flags & dpkt.tcp.TH_SYN) != 0)
            tcp_ack = int((tcp.flags & dpkt.tcp.TH_ACK) != 0)
            tcp_fin = int((tcp.flags & dpkt.tcp.TH_FIN) != 0)
            tcp_rst = int((tcp.flags & dpkt.tcp.TH_RST) != 0)
        elif isinstance(l4_payload, dpkt.udp.UDP):
            l4_udp = 1
            udp = l4_payload
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
        elif isinstance(l4_payload, (dpkt.icmp.ICMP, getattr(dpkt, "icmp6", object))):
            l4_icmp = 1
        else:
            pass

        flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{int(l4_tcp)*6+int(l4_udp)*17}"
        flow_hash = _safe_hash32(flow_key)

        pf = PacketFeatures(
            length=length,
            l4_tcp=l4_tcp,
            l4_udp=l4_udp,
            l4_icmp=l4_icmp,
            direction_out=-1,  # to be set later once we know device ip
            src_port=src_port,
            dst_port=dst_port,
            tcp_syn=tcp_syn,
            tcp_ack=tcp_ack,
            tcp_fin=tcp_fin,
            tcp_rst=tcp_rst,
            flow_hash=flow_hash,
        )
        return pf, {"src_ip": src_ip or "", "dst_ip": dst_ip or ""}
    except Exception:
        # Corrupt/unsupported frame: return minimal features
        pf = PacketFeatures(
            length=length,
            l4_tcp=0,
            l4_udp=0,
            l4_icmp=0,
            direction_out=-1,
            src_port=0,
            dst_port=0,
            tcp_syn=0,
            tcp_ack=0,
            tcp_fin=0,
            tcp_rst=0,
            flow_hash=0,
        )
        return pf, {"src_ip": "", "dst_ip": ""}


def guess_device_ip(packets_meta: List[Dict[str, str]]) -> Optional[str]:
    """Heuristic: pick the most frequent private source IP as device IP.

    If none, pick the most frequent source IP overall.
    """
    from collections import Counter

    srcs = [m.get("src_ip", "") for m in packets_meta if m.get("src_ip")]
    if not srcs:
        return None
    priv = [s for s in srcs if _is_private_ip(s)]
    c = Counter(priv or srcs)
    return c.most_common(1)[0][0]


def _direction_flag(src_ip: str, device_ip: Optional[str]) -> int:
    if not device_ip or not src_ip:
        return -1
    return int(src_ip == device_ip)


def parse_pcap_to_features(
    pcap_path: str | Path,
    cache_dir: Optional[str | Path] = None,
    device_ip: Optional[str] = None,
    guess_device: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Parse a pcap/pcapng into a numeric feature matrix [N, F].

    Args:
        pcap_path: path to pcap/pcapng file
        cache_dir: optional directory to cache parsed npz by file hash+mtime
        device_ip: optional known device IP to compute direction
        guess_device: if True and device_ip is None, guess from traffic

    Returns:
        features: ndarray [N, F]
        feature_names: list of names aligned with columns
    """
    pcap_path = str(pcap_path)
    path = Path(pcap_path)
    assert path.exists(), f"pcap not found: {pcap_path}"

    feature_names = PacketFeatures.feature_names() + [
        "iadelta",  # inter-arrival delta seconds
    ]

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig = f"{path.resolve()}::{path.stat().st_mtime_ns}"
        h = hashlib.blake2b(sig.encode("utf-8"), digest_size=8).hexdigest()
        cache_file = cache_dir / f"{h}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            feats = data["feats"]
            # Sanity check expected shape
            if feats.ndim == 2 and feats.shape[1] == len(feature_names):
                return feats, feature_names

    # Parse raw packets
    feats: List[PacketFeatures] = []
    metas: List[Dict[str, str]] = []
    timestamps: List[float] = []
    for ts, buf in _read_pcap(path):
        pf, meta = _parse_packet(ts, buf)
        feats.append(pf)
        metas.append(meta)
        timestamps.append(ts)

    # Direction inference
    dev_ip = device_ip or (guess_device and guess_device_ip(metas)) or None
    if dev_ip:
        print(f"Using device IP: {dev_ip} for {path.name}")
    else:
        print(f"Warning: Could not determine device IP for {path.name}. Direction will be -1.")

    for pf, meta in zip(feats, metas):
        pf.direction_out = _direction_flag(meta.get("src_ip", ""), dev_ip)

    # Convert to numpy
    rows = []
    prev_ts = None
    for pf, ts in zip(feats, timestamps):
        iadelta = 0.0 if prev_ts is None else max(0.0, ts - prev_ts)
        prev_ts = ts
        rows.append(
            [
                float(pf.length),
                float(pf.l4_tcp),
                float(pf.l4_udp),
                float(pf.l4_icmp),
                float(pf.direction_out),
                float(pf.src_port),
                float(pf.dst_port),
                float(pf.tcp_syn),
                float(pf.tcp_ack),
                float(pf.tcp_fin),
                float(pf.tcp_rst),
                float(pf.flow_hash),
                float(iadelta),
            ]
        )

    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(feature_names)), dtype=np.float32)

    if cache_file is not None:
        try:
            np.savez_compressed(cache_file, feats=arr)
        except Exception:
            pass

    return arr, feature_names


def read_labels_json(json_path: str | Path) -> np.ndarray:
    """Read JSON containing an array of packet numbers that are location requests.

    Wireshark is 1-based; convert to 0-based indices and to a dense 0/1 vector of length N lazily in Dataset.
    Here we only return a sorted array of zero-based indices.
    """
    path = Path(json_path)
    assert path.exists(), f"labels json not found: {json_path}"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Labels JSON must be a list of packet numbers")
    idxs = []
    for x in data:
        try:
            i = int(x) - 1
            if i >= 0:
                idxs.append(i)
        except Exception:
            continue
    idxs = sorted(set(idxs))
    return np.asarray(idxs, dtype=np.int64)
