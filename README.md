# RankCv-v1.2 — AI-Powered CV Ranking & Candidate Recommendation System

**🎥 Watch Live Demo First:**

[Watch RankCV Live Demo](https://www.linkedin.com/posts/dikachi-baron-a4a380356_python-flask-ai-activity-7464612973546975232-VA-H?utm_medium=member_android&rcm=ACoAAFiwEdoBhQHM9RGHGnevgOcCk1gtXoCOlv8&utm_source=chatgpt.com)

---

RankCV is a Flask-based AI-powered hiring assistant designed to simplify and improve recruitment by automatically analyzing, ranking, and recommending candidates from uploaded CVs.

Recruiters and hiring managers often spend hours manually reviewing hundreds or even thousands of resumes. This process can become slow, repetitive, and inconsistent, especially when handling large applicant volumes.

RankCV solves this problem by using Artificial Intelligence, semantic matching, and intelligent candidate analysis to compare CVs against job descriptions and automatically rank candidates from strongest to weakest match.

Instead of spending time manually screening candidates, recruiters receive organized, explainable, and data-driven recommendations instantly.

---

# 🚀 Core Features

## Intelligent Candidate Ranking

* Upload multiple CVs simultaneously
* Enter any job description
* Automatically rank candidates from highest match to lowest match
* Score candidates using AI semantic similarity

---

## AI-Powered Candidate Analysis

Each candidate includes:

* Match percentage score
* Strengths analysis
* Weaknesses analysis
* Skill comparison
* Experience evaluation
* Missing skills detection
* Suggested interview focus areas
* Recruiter-style AI assessment

---

## User Authentication System

Secure user account features:

* Registration
* Login
* Logout
* Password hashing
* Session management

---

## User Profile Dashboard

Users can:

* Upload profile pictures
* Monitor active subscription plans
* Track subscription dates
* View session usage
* Manage account information

---

## Subscription Plans

RankCV supports multiple plans:

| Plan    | CV/session | Daily Sessions |
| ------- | ---------- | -------------- |
| Free    | 10         | 1              |
| Basic   | 25         | 2              |
| Pro     | 50         | 3              |
| Premium | Unlimited  | Unlimited      |

---

## Ranking Dashboard

After processing CVs, users can see:

* Total uploaded CVs
* Strong matches
* Moderate matches
* Weak matches
* Candidate ranking leaderboard
* Candidate profile summaries

---

## Candidate Detail View

Clicking a candidate displays:

### Profile Summary

* Experience level
* Education
* Previous roles
* Companies worked for
* Skills matched
* Missing skills
* Areas for improvement

### Example

**Candidate Score: 82%**

**Why ranked highly?**

✓ 8+ years experience
✓ Strong Python background
✓ React expertise
✓ REST API development
✓ Senior Software Engineering experience

---

## Smart Settings System

Users can customize ranking behavior:

* Minimum experience
* Maximum experience
* Experience filters
* Score display
* CV limits
* Ranking preferences

---

## Ranking History

RankCV stores previous ranking sessions:

* Job title
* Top candidate
* Date created
* Number of CVs processed
* Historical results

---

# 🧠 AI Ranking Workflow

## Step 1

User uploads CVs and enters a job description.

## Step 2

System validates subscription limits.

## Step 3

PDF CVs are processed:

**Primary extraction**

* pdfplumber

**Fallback**

* PyPDF2

---

## Step 4

CV parser extracts:

* Candidate name
* Email
* Phone number
* Skills
* Experience
* Education
* Previous roles
* Companies
* Profile summary

---

## Step 5

Semantic AI scoring begins.

### Ranking Formula

Final Score:

* 60% Semantic Similarity
* 40% Skill Match Bonus

No CV receives a score below 10%.

---

## Step 6

AI explanation engine generates:

### Strengths

* Skills matched
* Experience advantages
* Education highlights

### Weaknesses

* Missing skills
* Experience gaps
* Qualification limitations

---

## Step 7

Results are sorted automatically:

**Strong Match → Moderate Match → Weak Match**

---

## Step 8

Candidate insights are enhanced using Google Gemini AI.

Recruiters receive:

* Candidate narrative
* Key highlights
* Concerns
* Interview recommendations

---

# 🛠 Technology Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| Backend          | Flask                 |
| Database         | MySQL                 |
| ORM              | Flask SQLAlchemy      |
| Semantic Search  | sentence-transformers |
| AI Model         | Google Gemini Flash   |
| Frontend         | Jinja2 + Bootstrap    |
| Authentication   | Flask Session         |
| PDF Processing   | pdfplumber + PyPDF2   |
| Machine Learning | Scikit-learn          |
| Email            | Gmail SMTP            |

---

# 📂 Project Structure

```bash
RankCv/
│
├── app.py
├── requirements.txt
├── RankCv_db.sql
│
├── static/
│   ├── css/
│   └── uploads/
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── history.html
│   ├── settings.html
│   ├── pricing.html
│   ├── faq.html
│   ├── changelog.html
│   ├── contact.html
│   └── how_it_works.html
```

---

# ⚡ Installation

### Clone repository

```bash
git clone https://github.com/Dikachi32/RankCV.git

cd RankCV
```

### Create virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install greenlet
pip install -r requirements.txt
```

### Setup database

```bash
mysql -u root -p < RankCv_db.sql
```

### Run project

```bash
python app.py
```

Visit:

```bash
http://localhost:5000
```

---

# Future Improvements

* Export results to PDF and Excel
* Candidate email integration
* Stripe payment system
* Multi-language CV support
* Admin analytics dashboard
* ATS API integration
* Bias-free anonymous CV mode
* Custom ranking weights
* Advanced recruiter analytics

---

# 👨‍💻 Author

**Dikachi Baron**

Email: [francisonyedikachiagwu@gmail.com](mailto:francisonyedikachiagwu@gmail.com)

X: @Baron_dikachi

LinkedIn: Dikachi Baron

Built with Flask, AI, MySQL, and Google Gemini.

Made with ❤️ in Nigeria
