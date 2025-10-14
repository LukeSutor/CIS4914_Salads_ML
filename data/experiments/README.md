# Experiment Data Collection – Find My Packet Capture Protocol

This document explains how to collect, organize, and share packet capture (pcap) files for all location-sharing experiments with Apple Find My, as performed in this project. Follow these steps for any new experiment to guarantee consistent and reproducible results.

***

## 1. Devices and Software Needed
- **MacBook** with:
  - Wireshark installed
  - Command line tools (`xcode-select --install`)
  - Access to Terminal
- **Two iPhones:**
  - One as "surveilled" device (sharing its location)
  - One as "surveilling" device (actively querying or viewing location)
- **USB/USB-C** Connector Cables

***

## 2. Pre-Experiment Setup

- Ensure **Wireshark** is installed on your MacBook (https://www.wireshark.org/download.html).
- On both iPhones:
  - Enable location sharing with each other in the Find My app (‘Share My Location’ and specify the other phone to share with).
  - Make sure both devices are signed into iCloud and have granted necessary permissions.

***

## 3. Packet Capture Procedure (Step-by-Step)

1. **Connect Target iPhone (e.g., the one being monitored) to MacBook via USB.**
   - Unlock the iPhone.
   - Tap "Trust This Computer" if prompted.

2. **Find the iPhone’s UDID:**
   - Open Finder, select the iPhone under Locations.
   - Click the Serial Number label until the UDID appears. Copy it.

3. **Enable Remote Virtual Interface (RVI) in Terminal:**
   - Open Terminal.
   - Start the Apple background service if needed:
     ```
     sudo launchctl load -w /Library/Apple/System/Library/LaunchDaemons/com.apple.rpmuxd.plist
     ```
   - Create the remote interface (replace `<UDID>` with your phone's real UDID):
     ```
     sudo rvictl -s <UDID>
     ```
   - You should see something like: `Starting device <UDID> [SUCCEEDED]`. A new interface (rvi0) is now available.

4. **Open Wireshark and Start Capture:**
   - Launch Wireshark on your MacBook.
   - Select the interface named `rvi0`.
   - Click the blue shark fin (start button) to begin recording packets.

5. **Set Up and Run Experiment Scenario:**
   - Decide which phone is performing which role (surveilled or surveilling).
   - Set scenario parameters (e.g., idle, query once per minute, query twice per minute, etc.).
   - For *active querying*: On the surveilling device, open Find My, select the other device, and trigger location queries at the experimental timing (for example, tap/refresh once per minute).
   - For *idle*: Simply leave Find My open or in the background as specified by scenario.
   - **Let the experiment run for at least 10 minutes per scenario.**
   - Use a timer to standardize query intervals.

6. **Finish and Save the Capture:**
   - After the scenario completes, stop the capture in Wireshark (red square button).
   - Save the capture file as `.pcapng` with a clear name (e.g., `bluey_idle_connected.pcapng` or `ultramarine_once_per_minute.pcapng`).
   - Move the file to `data/experiments/` in your local repo folder.

7. **Repeat Above for Each Experiment Scenario.**
   - Examples:
     - Surveilled idle (no queries)
     - Surveilled queried once per minute
     - Surveilled queried twice per minute
     - Surveilling device actively querying another’s location

***

## 4. Large .pcapng Files

```
Files that are too big to upload directly can be added here:
- once_per_min.pcapng: https://drive.google.com/file/d/1nJPygSLMIj59uz4sFGDf-ruBrAHq8hO5/view?usp=sharing
- active_queries.pcapng: https://drive.google.com/file/d/1oPevWx53oNe6scpak5Q1e8GMf2DL1eki/view?usp=sharing
```

***

## 6. Notes
- Each experiment should take at least 10 minutes.
- Confirm network traffic is captured by opening the .pcapng file in Wireshark and checking packet activity.
- Keep a log of scenario timings, roles, and any issues during capture for accurate labeling later.

***
