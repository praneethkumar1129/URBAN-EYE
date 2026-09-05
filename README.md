# URBAN-EYE

## AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet

**SIH 2026 — Problem Statement 26124**  
**Organization: Bharat Electronics Limited (BEL)**

URBAN-EYE transforms public transport buses into mobile urban sensing units.

The platform uses multi-camera video, edge AI, GPS, centralized event processing, geospatial analytics, and a command-center dashboard to detect and analyze urban road and traffic conditions.

## Architecture

The system follows a fixed edge-first architecture:

- On-Bus Edge AI System
- Secure Event-Only Transmission
- Centralized Cloud Backend
- PostgreSQL + PostGIS
- Analytics & AI Services
- File/Evidence Storage
- Notification & Ticketing
- Command Center GIS Dashboard
- Datasets & Model Training
- Cloud-Assisted Development and Training

## Core Capabilities

- Road defect detection
- Vehicle detection and tracking
- Traffic congestion analysis
- ANPR and OCR
- Pedestrian safety analysis
- Dangerous-driving event detection
- Multi-bus event correlation
- Geo-tagged evidence
- Road-condition heatmaps
- Route and traffic analytics
- Predictive maintenance
- Human-in-the-loop verification
- Maintenance ticket management
- Urban digital memory
- Privacy-aware event processing

## Development Environment

Development is performed primarily on a local Windows system.

Heavy model training and other compute-intensive workloads may use AWS or Google Cloud when required.

## Project Structure

```text
URBAN-EYE/
├── edge/
├── backend/
├── frontend/
├── models/
├── datasets/
├── evidence/
├── scripts/
├── tests/
├── docker/
└── docs/
