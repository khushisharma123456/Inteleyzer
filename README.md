# Inteleyzer - Pharmacovigilance Platform

## 🎯 Overview
Inteleyzer is a comprehensive pharmacovigilance system for monitoring and managing adverse drug reactions (ADRs). It supports three user roles: **Pharmaceutical Companies**, **Doctors**, and **Local Pharmacies**.

## 📁 Clean Project Structure

```
Novartis/
├── app.py                    # Main Flask application (PORT: 5000)
├── models.py                 # Database models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── instance/                 # Database files
│   ├── inteleyzer.db        # Main application database
│   └── pv_database.db       # PV backend database
│
├── static/                   # Static assets (CSS, JS, images)
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── auth.js
│       ├── sidebar.js
│       └── pharma.js
│
├── templates/                # HTML templates
│   ├── index.html           # Landing page
│   ├── login.html           # Login page
│   ├── signup.html          # Registration page
│   ├── doctor/              # Doctor dashboard pages
│   │   ├── dashboard.html
│   │   ├── patients.html
│   │   └── alerts.html
│   ├── pharma/              # Pharma company pages
│   │   ├── dashboard.html
│   │   ├── drugs.html
│   │   ├── reports.html
│   │   └── analysis.html
│   └── pharmacy/            # Local pharmacy pages
│       ├── dashboard.html
│       ├── reports.html
│       ├── report.html
│       └── alerts.html
│
├── utils/                    # Utility scripts
│   └── populate_complete_data.py  # Database population script
│
├── docs/                     # Documentation & credentials
│   ├── ALL_LOGIN_CREDENTIALS.md
│   ├── ANALYSIS_FEATURE_DOCUMENTATION.md
│   ├── PHARMA_DATABASE_INFO.md
│   └── complete_database.xlsx
│
└── pv_backend/              # Separate PV system (PORT: 5001)
    ├── app.py
    ├── routes/
    ├── services/
    └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Application
Open your browser and navigate to: 

## 👥 User Roles & Features

### 🏢 Pharmaceutical Companies
- Monitor ADR reports for their drugs
- View analytics and risk assessments
- Send safety alerts to doctors and pharmacies
- Manage drug portfolio

### 👨‍⚕️ Doctors
- File ADR reports for patients
- View assigned patient cases
- Receive safety alerts
- Track patient outcomes

### 💊 Local Pharmacies
- Report ADRs observed at point of sale
- View their submitted reports
- Receive drug safety alerts
- Monitor dispensing statistics

## 🔐 Login Credentials

Check `/docs/ALL_LOGIN_CREDENTIALS.md` for complete list of test accounts.

**Sample Accounts:**
- **Pharma**: admin@novartis.com / novartis2024
- **Doctor**: emily.chen@hospital.com / doctor123
- **Pharmacy**: downtown@cvs-pharmacy.com / pharmacy123

## 📊 Database

### Main Database (`instance/inteleyzer.db`)
Contains:
- 8 Pharmaceutical Companies
- 10 Doctors (various specialties)
- 10 Local Pharmacies
- 27 Drugs
- 200 ADR Reports (120 from doctors, 80 from pharmacies)
- 60 Safety Alerts

### Population Script
To regenerate test data:
```bash
python utils/populate_complete_data.py
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Components**: Lucide Icons, Chart.js
- **Authentication**: Session-based

## 📂 What Was Cleaned Up

**Removed Duplicates:**
- ❌ Old `backend/` folder (consolidated to root)
- ❌ Duplicate doctor/hospital/pharmacy HTML folders
- ❌ Scattered CSS/JS files
- ❌ Multiple instance folders

**Organized Structure:**
- ✅ Single `app.py` entry point
- ✅ Centralized `static/` and `templates/`
- ✅ Dedicated `utils/` for scripts
- ✅ `docs/` for documentation
- ✅ Clean separation from `pv_backend/` (separate system)

## 🔄 Migration Notes

**Old Structure** → **New Structure**
- `backend/app.py` → `app.py`
- `backend/static/` → `static/`
- `backend/templates/` → `templates/`
- `backend/models.py` → `models.py`
- `backend/populate_complete_data.py` → `utils/populate_complete_data.py`
- `backend/*.md` → `docs/*.md`

**All paths updated automatically!** ✨

## 🎨 Features

- 📊 **Real-time Analytics** - Drug safety metrics and trends
- 🚨 **Alert System** - Push notifications for critical events
- 📈 **Risk Assessment** - AI-powered drug risk evaluation
- 👥 **Multi-role Support** - Tailored dashboards per user type
- 📱 **Responsive Design** - Works on all devices
- 🔒 **Secure Authentication** - Session-based login system

---

**Last Updated**: January 27, 2026  
**Version**: 2.0 (Clean & Reorganized)
