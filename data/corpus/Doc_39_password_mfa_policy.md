---
doc_id: Doc_39
title: Password and MFA Policy
category: it_devices
effective_date: 2026-04-01
status: current
---

# Password and MFA Policy

## §1 Password Requirements
The password requirements at Halcyon Labs India Pvt. Ltd. apply to every corporate account, including SSO, email, VPN and ServiceHub.

- Minimum length is 14 characters.
- You may not reuse any of your last 10 passwords.
- Passwords must be rotated every 180 days; ServiceHub sends reminders 14 days and 3 days before expiry.
- Passphrases made of several unrelated words are encouraged because they are easier to remember.
- Never include your name, employee ID or the company name in a password.

Change your password through the self-service portal, which also updates the saved credentials for GlobalConnect after the next sign-in.

## §2 Multi-Factor Authentication
Multi-factor authentication (MFA) is required for all corporate sign-ins, and an authenticator app push is mandatory as the primary method. When the push arrives, check that the location and application shown match what you are doing, then approve it. Deny any push you did not trigger and report it to security@halcyonlabs.example straight away.

SMS codes are allowed only as a fallback, for example when you have changed phones and have not yet re-registered the authenticator app. To move the app to a new phone, register the new device from the self-service portal before you wipe the old one.

## §3 Account Lockout
Account lockout is triggered after 5 failed sign-in attempts, and the account stays locked for 30 minutes. The lockout clears automatically when the period ends.

If you cannot wait, contact the helpdesk on extension 4357 or through ServiceHub chat. The helpdesk unlocks the account only after identity verification, which includes your employee ID, your manager's name and an approval sent to your registered mobile number. Repeated lockouts are reviewed by the Security team, because they can indicate a password-guessing attempt.

## §4 Password Managers
For storing passwords, only the company-provided password vault is approved at Halcyon Labs. Install it from Software Center and sign in with SSO plus MFA.

- Do not keep work passwords in browsers, spreadsheets, notes apps or personal password managers.
- Do not paste passwords into chat or email.
- Use the vault's generator to create unique passwords of at least 14 characters for tools that are not covered by SSO.
- Shared team credentials belong in a shared vault folder owned by the team lead.

Vault contents are backed up and can be recovered by the helpdesk if you forget your master credentials.
