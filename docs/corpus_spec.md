# Demo Corpus Specification

The hackathon brief states that a corpus is provided, but none was distributed with the
theme guide we received. DuplexRAG is corpus-agnostic (drop any `.md/.txt/.pdf/.jsonl`
documents into `data/corpus/` and re-index), so for development and benchmarking we
authored a **synthetic enterprise knowledge base** for a fictional company,
*Halcyon Labs India Pvt. Ltd.* All names, venues, vendors, prices and policies are
fictional. The documents were drafted with AI assistance from this specification and
then validated by a script (`scripts/validate_corpus.py`).

The corpus deliberately mirrors the scenarios in the Theme 4 guide (customer workshop
venues in Pune, travel reimbursement with late-booking and international exceptions)
and contains engineered hard cases:

| Hard case | Where |
|---|---|
| Missing evidence (must trigger an uncertainty flag) | Doc_04 and Doc_10 have **no catering information** |
| Superseded / conflicting policy | Doc_23 (Travel Policy 2024, superseded by Doc_15) |
| Redundant evidence (dedup + fusion) | Doc_02 directory repeats venue capacities |
| Near-duplicate topics across domains | Doc_16 vs Doc_22 (claims), Doc_41 vs Doc_42 (device issues) |

## Document format

```
---
doc_id: Doc_03
title: Riverside Conference Centre, Pune - Venue Fact Sheet
category: events_venues
effective_date: 2026-04-01
status: current            # or: superseded (with superseded_by: Doc_15)
---

# Riverside Conference Centre, Pune - Venue Fact Sheet

## §1 Overview and Location
...
## §2 Capacity and Room Layouts
...
```

Citations use the form `Doc_03 §2`.

## Document list

### Events and venues (Doc_01 - Doc_14)
Doc_01 Event Planning Policy - Doc_02 Preferred Venue Directory - Doc_03 Riverside
Conference Centre, Pune - Doc_04 Hinjewadi Tech Park Training Suites, Pune (no catering
info) - Doc_05 The Banyan Court, Koregaon Park, Pune - Doc_06 Whitefield Innovation Hub,
Bengaluru - Doc_07 Cubbon Heritage Hall, Bengaluru - Doc_08 HITEC City Convention Rooms,
Hyderabad - Doc_09 OMR Knowledge Centre, Chennai - Doc_10 Sector 62 Learning Centre,
Noida (no catering info) - Doc_11 Approved Catering Vendors - Doc_12 Event Safety and
Insurance Requirements - Doc_13 Customer Workshop Playbook - Doc_14 Team Offsite and
Outing Guidelines

### Travel and expense (Doc_15 - Doc_26)
Doc_15 Global Travel Policy (current) - Doc_16 Travel Reimbursement Rules - Doc_17 Per
Diem and Meal Allowances - Doc_18 Hotel Accommodation Caps - Doc_19 Ground Transport and
Mileage - Doc_20 International Travel: Visa, Insurance and Forex - Doc_21 Corporate Card
Policy - Doc_22 Expense Claim Process (SpendWise) - Doc_23 Travel Policy 2024
(superseded) - Doc_24 Relocation Assistance - Doc_25 Client Entertainment and Gifts -
Doc_26 Travel Safety and Emergency Support

### HR and workplace (Doc_27 - Doc_36)
Doc_27 Leave Policy - Doc_28 Parental Leave - Doc_29 Hybrid Work Policy - Doc_30
Learning and Development Reimbursement - Doc_31 Wellness Benefits - Doc_32 Holiday
Calendar 2026 - Doc_33 Onboarding Guide - Doc_34 Code of Conduct and POSH - Doc_35
Internal Job Postings - Doc_36 Performance Review Cycle

### IT and devices (Doc_37 - Doc_46)
Doc_37 Laptop Provisioning and Refresh - Doc_38 VPN and Remote Access - Doc_39 Password
and MFA Policy - Doc_40 BYOD and Mobile Device Policy - Doc_41 Smartphone
Troubleshooting: Battery Drain - Doc_42 Smartphone Troubleshooting: Display Flicker -
Doc_43 Software Requests and Licensing - Doc_44 Security Incident Reporting - Doc_45
Meeting Room AV and Video Conferencing - Doc_46 IT Helpdesk Service Levels
