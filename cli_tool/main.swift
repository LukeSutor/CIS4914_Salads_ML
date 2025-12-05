#!/usr/bin/env swift
import Foundation

guard getuid() == 0 else {
    print("Root permissions required (run with sudo). Exiting.")
    exit(1)
}

struct DataPacket {
    let rawData: Data               // Original packet data
    let timestamp: String           // Time the packet was captured
    let protocolName: String        // TCP, UDP, ICMP, etc.
    let srcAddress: String          // Source IP
    let srcPort: Int?               // Source port (nil if not TCP/UDP)
    let dstAddress: String          // Destination IP
    let dstPort: Int?               // Destination port (nil if not TCP/UDP)
    let payload: Data
}

func parseLine(_ line: String) -> DataPacket? {
    // Split into words
    let comps = line.split(separator: " ")
    guard comps.count > 5 else { return nil }

    // Detect protocol type
    let protoField = comps[1]
    let isV4 = protoField == "IP"
    let isV6 = protoField == "IP6"
    guard isV4 || isV6 else { return nil }

    let timestamp = String(comps[0])
    let protocolName = String(protoField)

    // IPv4: srcIP.srcPort > dstIP.dstPort:
    // IPv6: srcIPv6.port > dstIPv6.port:
    let srcField = comps[2]
    let dstField = comps[4].trimmingCharacters(in: CharacterSet(charactersIn: ":"))

    // For both, parse src and dst IP/Port
    func splitIPAndPort(_ field: Substring) -> (String, Int?) {
        // For IPv4: 192.168.1.1.54321
        // For IPv6: 2603:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx.12345
        if let dotIdx = field.lastIndex(of: ".") {
            let ipPart = field[..<dotIdx]
            let portPart = field[field.index(after: dotIdx)...]
            return (String(ipPart), Int(portPart))
        } else {
            return (String(field), nil)
        }
    }
    let (srcAddress, srcPort) = splitIPAndPort(srcField)
    let (dstAddress, dstPort) = splitIPAndPort(Substring(dstField))

    // Attempt to extract length field
    var lengthField: Int = 0
    for i in 0..<(comps.count-1) {
        if comps[i].starts(with: "length") {
            if comps[i] == "length", let l = Int(comps[i+1]) {
                lengthField = l
            } else if let l = Int(comps[i].dropFirst("length".count)) {
                lengthField = l
            }
            break
        }
    }
    let payload = Data()  // Optionally, extract more data if needed
    let rawData = line.data(using: .utf8) ?? Data()

    return DataPacket(rawData: rawData, timestamp: timestamp, protocolName: protocolName, srcAddress: srcAddress, srcPort: srcPort, dstAddress: dstAddress, dstPort: dstPort, payload: payload)
}

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/sbin/tcpdump")
process.arguments = ["-i", "rvi0", "-n", "-l", "-p"]

let pipe = Pipe()
process.standardOutput = pipe
try process.run()

pipe.fileHandleForReading.readabilityHandler = { handle in
    guard let str = String(data: handle.availableData, encoding: .utf8), !str.isEmpty else { return }
    let lines = str.split(separator: "\n", omittingEmptySubsequences: true).map { String($0) }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    for line in lines {
        if let pkt = parseLine(line) {
            struct Out: Encodable {
                let timestamp: String
                let protocolName: String
                let srcAddress: String
                let srcPort: Int?
                let dstAddress: String
                let dstPort: Int?
                let raw: String
            }
            let out = Out(timestamp: pkt.timestamp, protocolName: pkt.protocolName, srcAddress: pkt.srcAddress, srcPort: pkt.srcPort, dstAddress: pkt.dstAddress, dstPort: pkt.dstPort, raw: String(data: pkt.rawData, encoding: .utf8) ?? "")
            if let json = try? encoder.encode(out), let s = String(data: json, encoding: .utf8) {
                print(s)
            } else {
                print("Parsed: \(pkt)")
            }
        } else {
            print("Unparsed: \(line)")
        }
    }
}

RunLoop.main.run()
