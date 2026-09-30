---
doc_id: Doc_38
title: VPN and Remote Access
category: it_devices
effective_date: 2026-04-01
status: current
---

# VPN and Remote Access

## §1 GlobalConnect VPN Overview
GlobalConnect is the company VPN client at Halcyon Labs India Pvt. Ltd., and it is mandatory for reaching internal systems whenever you are off the office network. This covers the intranet, code repositories, ServiceHub admin views, internal dashboards and file shares. Cloud applications that sit behind SSO, such as email and chat, can be used without the VPN, but any system hosted on the internal network cannot. When you are in an office, connect to the corporate Wi-Fi or a wired port and leave GlobalConnect disconnected.

## §2 Setup
To set up GlobalConnect on a company laptop, install the client from Software Center and sign in with your company identity.

1. Open Software Center and search for "GlobalConnect".
2. Click Install and wait for the confirmation message.
3. Launch GlobalConnect and enter the gateway address shown on the ServiceHub knowledge page.
4. Sign in with SSO, then approve the multi-factor authentication push on your authenticator app.
5. Confirm that the status shows "Connected" before opening internal sites.

The device certificate is issued automatically during the first sign-in.

## §3 Common Errors
For GlobalConnect connection failures, match the error message to the fix below.

- Error 809: a firewall is blocking UDP 500/4500, which is common on hotel and home routers. Open GlobalConnect > Settings > Protocol and switch the client to TCP mode, then reconnect.
- Error 691: your password has expired. Reset it through the self-service portal, then sign in to GlobalConnect again with the new password.
- "Certificate expired": the device certificate is out of date. Re-enrol the device from Software Center by running the "GlobalConnect Certificate Enrol" package, then restart the client.

If the error persists after these steps, raise a ServiceHub ticket with the error text and a screenshot of the client log.

## §4 Usage Rules
The usage rules for GlobalConnect apply to every employee and contractor who connects to internal systems remotely.

- No split tunnelling: all traffic must pass through the VPN while it is connected, and the client is configured to block bypass routes.
- Never use the VPN on public or shared computers, such as internet cafes, hotel business centres or a friend's laptop.
- Do not share your credentials or authenticator approvals with anyone.
- Disconnect when you no longer need internal access, and always before leaving your laptop unattended.

Breaches of these rules are reported to the Security team and may lead to access suspension.
