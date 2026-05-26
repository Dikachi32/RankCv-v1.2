from flask import Flask, render_template, request, redirect, flash, jsonify, session, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timezone
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import os
import json
import re
import io
import logging

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================================================================
# PDF EXTRACTION
# =============================================================================
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    try:
        import PyPDF2
    except ImportError:
        pass

# =============================================================================
# SEMANTIC SCORING — sentence-transformers (loads once into RAM, fast after)
# =============================================================================
try:
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    from sentence_transformers import SentenceTransformer, util as st_util
    import numpy as np
    print("[EMBEDDINGS] Loading all-MiniLM-L6-v2 (one-time load, ~5 sec)...")
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_EMBEDDINGS  = True
    print("[EMBEDDINGS] Model loaded and ready!")
except ImportError:
    HAS_EMBEDDINGS  = False
    EMBEDDING_MODEL = None
    print("[EMBEDDINGS] sentence-transformers not installed — using TF-IDF fallback.")
    print("             Run: pip install sentence-transformers")

# =============================================================================
# TF-IDF — fallback only if sentence-transformers not available
# =============================================================================
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# GEMINI API — on-demand AI explanations when user clicks a candidate
# =============================================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyB-sdOi4S6GHtgwp9Ms_DZMsEBCynKv33s')
GEMINI_MODEL   = 'gemini-1.5-flash'

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_CLIENT    = genai.GenerativeModel(GEMINI_MODEL)
    GEMINI_AVAILABLE = True
    print("[GEMINI] Configured — Gemini Flash ready for on-demand explanations.")
except ImportError:
    GEMINI_AVAILABLE = False
    GEMINI_CLIENT    = None
    print("[GEMINI] Not installed. Run: pip install google-generativeai")
except Exception as e:
    GEMINI_AVAILABLE = False
    GEMINI_CLIENT    = None
    print(f"[GEMINI] Config error: {e}")

# =============================================================================
# REGEX CV PARSER — name, skills, experience, education (instant, no API)
# =============================================================================
class CVParser:
    SKILL_KEYWORDS = {
        'python','java','javascript','typescript','react','angular','vue',
        'node.js','nodejs','django','flask','fastapi','sql','mysql','postgresql',
        'mongodb','aws','azure','gcp','docker','kubernetes','jenkins',
        'git','github','gitlab','ci/cd','machine learning','deep learning',
        'tensorflow','pytorch','scikit-learn','pandas','numpy','matplotlib',
        'tableau','powerbi','power bi','excel','linux','bash','shell','rest api',
        'graphql','microservices','agile','scrum','jira','confluence',
        'html','css','sass','bootstrap','tailwind','jquery','php','laravel',
        'ruby','rails','go','golang','rust','c++','c#','.net','spring',
        'kafka','redis','elasticsearch','terraform','ansible','prometheus',
        'swift','kotlin','flutter','react native','opencv','nlp',
        'computer vision','data science','big data','hadoop','spark',
        'airflow','dbt','snowflake','looker','sap','oracle','salesforce',
        'figma','sketch','photoshop','illustrator','unity','leadership',
        'management','communication','problem solving','teamwork',
        'project management','data analysis','data visualization',
        'etl','business intelligence','analytics','seo','digital marketing',
        'ui/ux','user research','devops','site reliability','cloud architecture',
        'blockchain','artificial intelligence','neural networks',
        'natural language processing','generative ai','llm','prompt engineering',
        'api design','system design','unit testing','test automation',
        'cybersecurity','mobile development','ios','android',
        'html5','css3','express','next.js','svelte','spring boot',
        'streamlit','plotly','seaborn','selenium','playwright',
        'redux','webpack','jest','mocha','cypress','wordpress','shopify',
        'power automate','power apps','sharepoint','dynamics 365',
        'adobe xd','wireframing','prototyping','figma','sketch',
        'amazon web services','google cloud','azure devops','github actions',
        'continuous integration','continuous deployment','infrastructure as code',
    }

    NAME_SKIP = {
        'resume','cv','curriculum','vitae','email','phone','address',
        'linkedin','github','portfolio','objective','summary','experience',
        'education','skills','references','contact','www','http','profile',
        'about','career','overview','introduction','background','interests',
        'publications','certifications','awards','languages','technical',
        'university','college','school','institute','company','corporation',
        'professional','personal','information','details','page',
    }

    def __init__(self, text, filename=None):
        self.text       = text
        self.filename   = filename
        self.lines      = [l.strip() for l in text.split('\n') if l.strip()]
        self.text_lower = text.lower()

    def extract_name(self):
        # Strategy 1: scan first 15 lines for a name-shaped line
        for line in self.lines[:15]:
            if self._is_likely_name(line):
                return self._clean_name(line)
        # Strategy 2: explicit "Name:" label
        m = re.search(
            r'(?:name|full\s*name)\s*[:=]\s*([A-Za-z][a-zA-Z\-\.\']+(?:\s+[A-Za-z][a-zA-Z\-\.\']+){1,3})',
            self.text[:2000], re.IGNORECASE
        )
        if m:
            return self._clean_name(m.group(1))
        # Strategy 3: email prefix
        email = self.extract_email()
        if email:
            parts = re.split(r'[._\-]', email.split('@')[0])
            if len(parts) >= 2 and all(len(p) > 1 for p in parts[:2]):
                return ' '.join(p.capitalize() for p in parts[:2])
        # Strategy 4: filename
        if self.filename:
            n = self._name_from_filename(self.filename)
            if n:
                return n
        return 'Unknown Candidate'

    def _name_from_filename(self, fn):
        base = re.sub(r'\.pdf$', '', fn, flags=re.IGNORECASE)
        base = re.sub(r'(_cv|_resume|_application|_profile|_vita)$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'[_\-\.]+', ' ', base).strip()
        words = [w.capitalize() for w in base.split() if w.isalpha()]
        return ' '.join(words) if 2 <= len(words) <= 4 else None

    def _is_likely_name(self, line):
        if not line or len(line) > 55 or len(line) < 3:
            return False
        words = line.split()
        if not (1 < len(words) <= 5):
            return False
        if any(c.isdigit() for c in line):
            return False
        if any(c in '!@#$%^&*()[]{}|;:<>?/`~"_=+\\@,' for c in line):
            return False
        ll = line.lower()
        if any(kw in ll.split() for kw in self.NAME_SKIP):
            return False
        alpha_words = [w for w in words if re.match(r"^[A-Za-z\-'\.]+$", w)]
        if not alpha_words or len(alpha_words) < len(words) * 0.8:
            return False
        cap = sum(1 for w in alpha_words if w[0].isupper())
        return cap >= len(alpha_words) * 0.5

    def _clean_name(self, name):
        name = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Eng\.?|Engr\.?)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*,?\s*(Jr\.?|Sr\.?|II|III|IV|MBA|PhD|BSc|MSc|MD)\s*$', '', name, flags=re.IGNORECASE)
        name = ' '.join(name.split())
        name = ' '.join(w.capitalize() for w in name.split())
        return name.strip() or 'Unknown Candidate'

    def extract_email(self):
        m = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', self.text)
        return m.group(0) if m else None

    def extract_phone(self):
        m = re.search(r'(\+?\d[\d\s\-().]{7,}\d)', self.text)
        return m.group(0).strip() if m else None

    def extract_skills(self):
        found = []
        tl = self.text_lower
        for skill in self.SKILL_KEYWORDS:
            sl = skill.lower()
            try:
                if sl[0].isalnum() and sl[-1].isalnum():
                    pat = r'(?<!\w)' + re.escape(sl) + r'(?!\w)'
                elif sl[0].isalnum():
                    pat = r'(?<!\w)' + re.escape(sl)
                elif sl[-1].isalnum():
                    pat = re.escape(sl) + r'(?!\w)'
                else:
                    pat = re.escape(sl)
                if re.search(pat, tl):
                    found.append(skill)
            except re.error:
                continue
        return sorted(set(found))

    def extract_experience_years(self):
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in\s*)?(?:the\s*)?(?:industry|field|development|engineering|sector)',
            r'(?:over|more\s*than)\s*(\d+)\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:exp|experience)',
        ]
        vals = []
        for pat in patterns:
            for m in re.findall(pat, self.text_lower):
                try:
                    v = int(m)
                    if 0 < v < 50:
                        vals.append(v)
                except ValueError:
                    pass
        return max(vals) if vals else None

    def extract_education(self):
        edu_kw = [
            'bachelor','master','phd','doctorate','b.sc','m.sc','b.tech','m.tech',
            'mba','degree','diploma','university','college','institute','hnd','ond',
            'b.eng','m.eng','b.a.','m.a.','associate',
        ]
        edu = []
        for line in self.lines:
            ll = line.lower()
            if any(k in ll for k in edu_kw) and len(line) > 10:
                if line not in edu:
                    edu.append(line.strip())
        return edu[:3]

    def extract_job_titles(self):
        kw = [
            'engineer','developer','manager','analyst','architect','designer',
            'consultant','specialist','lead','director','scientist','researcher',
            'coordinator','administrator','executive','intern','trainee','officer',
            'supervisor','technician','programmer','devops','head of','vp of',
        ]
        titles = []
        for line in self.lines:
            ll = line.lower()
            if any(k in ll for k in kw):
                words = line.split()
                if 2 <= len(words) <= 9 and line not in titles:
                    titles.append(line.strip())
        return titles[:5]

    def extract_summary(self):
        kw = ['summary','objective','about','profile','overview','professional background']
        for i, line in enumerate(self.lines):
            if any(k in line.lower() for k in kw) and len(line) < 50:
                parts = []
                for j in range(i + 1, min(i + 5, len(self.lines))):
                    nl = self.lines[j].strip()
                    if 30 < len(nl) < 400:
                        parts.append(nl)
                    if len(parts) >= 2:
                        break
                if parts:
                    return ' '.join(parts)
        return ''

    def extract_companies(self):
        companies = []
        patterns = [
            r'(?:at|with|for)\s+([A-Z][A-Za-z0-9\s&,\.]+(?:Ltd\.?|Inc\.?|LLC|Corp\.?|Limited|GmbH|PLC)?)',
            r'([A-Z][A-Za-z0-9\s&]+(?:Ltd\.?|Inc\.?|LLC|Corp\.?|Limited))\s*(?:-|–|\|)',
        ]
        for pat in patterns:
            for m in re.findall(pat, self.text):
                c = m.strip()
                if 3 < len(c) < 60 and c not in companies:
                    companies.append(c)
        return companies[:4]

    def get_full_profile(self):
        return {
            'name':             self.extract_name(),
            'email':            self.extract_email(),
            'phone':            self.extract_phone(),
            'skills':           self.extract_skills(),
            'experience_years': self.extract_experience_years(),
            'education':        self.extract_education(),
            'job_titles':       self.extract_job_titles(),
            'companies':        self.extract_companies(),
            'summary':          self.extract_summary(),
            'raw_text_preview': self.text[:600],
        }

# =============================================================================
# SCORING ENGINE
# Primary:  sentence-transformers semantic similarity (accurate, understands meaning)
# Fallback: TF-IDF + keyword overlap (if sentence-transformers not installed)
# Both include a skill-match bonus so scores are realistic and never 0%
# =============================================================================
STOPWORDS = {
    'a','an','and','are','as','at','be','been','being','by','for','from',
    'has','have','had','he','her','him','his','how','i','in','is','it',
    'its','me','my','of','on','or','our','she','so','than','that','the',
    'their','them','then','there','these','they','this','those','to','too',
    'was','we','were','what','when','where','which','who','will','with',
    'would','you','your','am','shall','may','might','must','can','could',
    'about','after','also','any','both','but','each','even','just','like',
    'more','most','no','not','only','other','same','some','such','very',
}

def _keywords(text):
    words = re.findall(r'\b[a-z][a-z0-9+#.]{2,}\b', text.lower())
    return set(w for w in words if w not in STOPWORDS)

def _keyword_overlap(jd, cv):
    jd_kw = _keywords(jd)
    cv_kw = _keywords(cv)
    if not jd_kw:
        return 0.0
    return len(jd_kw & cv_kw) / len(jd_kw) * 100

def _skill_match_bonus(job_desc, profile):
    """
    Direct skill-keyword comparison between JD and parsed CV skills.
    Returns 0-40 bonus points.
    """
    jd_lower  = job_desc.lower()
    cv_skills = set(s.lower() for s in profile.get('skills', []))
    if not cv_skills:
        return 0.0
    job_skills = set()
    for skill in CVParser.SKILL_KEYWORDS:
        sl = skill.lower()
        try:
            if sl[0].isalnum() and sl[-1].isalnum():
                pat = r'(?<!\w)' + re.escape(sl) + r'(?!\w)'
            elif sl[0].isalnum():
                pat = r'(?<!\w)' + re.escape(sl)
            elif sl[-1].isalnum():
                pat = re.escape(sl) + r'(?!\w)'
            else:
                pat = re.escape(sl)
            if re.search(pat, jd_lower):
                job_skills.add(sl)
        except re.error:
            continue
    if not job_skills:
        return 0.0
    matched = len(job_skills & cv_skills)
    return (matched / len(job_skills)) * 40


def compute_scores(job_desc, cv_texts, parsed_profiles=None):
    """
    Master scoring function.
    Uses sentence-transformers if available (semantic, accurate).
    Falls back to TF-IDF if not installed.
    Minimum score = 10 (no CV ever shows 0%).
    """
    if not cv_texts:
        return []

    profiles = parsed_profiles or [{} for _ in cv_texts]

    # ── PRIMARY: Sentence-Transformers semantic scoring ──
    if HAS_EMBEDDINGS and EMBEDDING_MODEL is not None:
        try:
            print("[SCORING] Using semantic embeddings (sentence-transformers)...")
            jd_embedding  = EMBEDDING_MODEL.encode(job_desc, convert_to_tensor=True)
            scores = []
            for i, cv_text in enumerate(cv_texts):
                # Split CV into chunks so we catch the most relevant section
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cv_text) if len(s.strip()) > 20]
                if not sentences:
                    sentences = [cv_text]

                # Build chunks of ~120 words
                chunks, current, cur_len = [], [], 0
                for sent in sentences:
                    wc = len(sent.split())
                    if cur_len + wc > 120 and current:
                        chunks.append(' '.join(current))
                        current, cur_len = [sent], wc
                    else:
                        current.append(sent)
                        cur_len += wc
                if current:
                    chunks.append(' '.join(current))
                chunks = chunks[:20]  # cap at 20 chunks per CV

                if not chunks:
                    scores.append(10.0)
                    continue

                cv_embeddings = EMBEDDING_MODEL.encode(chunks, convert_to_tensor=True)
                sims          = st_util.cos_sim(jd_embedding, cv_embeddings)
                max_sim       = float(sims.max())   # best-matching chunk
                avg_top3      = float(sims.topk(min(3, sims.shape[1])).values.mean())
                semantic_score = (max_sim * 0.6 + avg_top3 * 0.4) * 100  # 0-100

                # Skill bonus — directly boosts score for matched skills
                skill_bonus   = _skill_match_bonus(job_desc, profiles[i])  # 0-40

                # Final blend: 60% semantic + 40% skill match
                raw   = (semantic_score * 0.60) + (skill_bonus * 0.40 * (100/40))
                final = round(min(max(raw, 10.0), 99.9), 1)
                scores.append(final)

            print(f"[SCORING] Semantic scoring complete. Scores: {scores}")
            return scores

        except Exception as e:
            print(f"[SCORING] Semantic scoring failed: {e} — falling back to TF-IDF")

    # ── FALLBACK: TF-IDF ──
    print("[SCORING] Using TF-IDF fallback...")
    if not HAS_SKLEARN:
        scores = []
        for i, t in enumerate(cv_texts):
            kw    = _keyword_overlap(job_desc, t)
            bonus = _skill_match_bonus(job_desc, profiles[i])
            scores.append(round(min(max(kw * 0.6 + bonus, 10), 99.9), 1))
        return scores

    try:
        docs = [job_desc] + cv_texts
        vec  = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=1,
        )
        mat    = vec.fit_transform(docs)
        jd_vec = mat[0]
        cv_mat = mat[1:]
        sims   = cosine_similarity(jd_vec, cv_mat).flatten()
        scores = []
        for i, sim in enumerate(sims):
            tfidf_score = float(sim) * 100
            kw_score    = _keyword_overlap(job_desc, cv_texts[i])
            skill_bonus = _skill_match_bonus(job_desc, profiles[i])
            raw   = (tfidf_score * 0.40) + (kw_score * 0.20) + (skill_bonus * (100/40) * 0.40)
            final = round(min(max(raw, 10.0), 99.9), 1)
            scores.append(final)
        return scores
    except Exception as e:
        print(f"[SCORING] TF-IDF error: {e}")
        scores = []
        for i, t in enumerate(cv_texts):
            kw    = _keyword_overlap(job_desc, t)
            bonus = _skill_match_bonus(job_desc, profiles[i])
            scores.append(round(min(max(kw + bonus, 10), 99.9), 1))
        return scores

# =============================================================================
# RULE-BASED EXPLANATION (instant, no API)
# =============================================================================
def rule_based_explanation(job_desc, profile, score):
    jd_lower  = job_desc.lower()
    cv_skills = set(s.lower() for s in profile.get('skills', []))

    job_skills = set()
    for skill in CVParser.SKILL_KEYWORDS:
        sl = skill.lower()
        try:
            if sl[0].isalnum() and sl[-1].isalnum():
                pat = r'(?<!\w)' + re.escape(sl) + r'(?!\w)'
            elif sl[0].isalnum():
                pat = r'(?<!\w)' + re.escape(sl)
            elif sl[-1].isalnum():
                pat = re.escape(sl) + r'(?!\w)'
            else:
                pat = re.escape(sl)
            if re.search(pat, jd_lower):
                job_skills.add(sl)
        except re.error:
            continue

    matched = sorted(job_skills & cv_skills)
    missing = sorted(job_skills - cv_skills)
    extra   = sorted(cv_skills - job_skills)

    strengths = []
    if matched:
        strengths.append(f"Matched key skills: {', '.join(s.title() for s in matched[:6])}")
    exp = profile.get('experience_years')
    if exp:
        strengths.append(
            f"Highly experienced: {exp}+ years" if exp >= 7 else
            f"Solid experience: {exp} years"    if exp >= 4 else
            f"Relevant experience: {exp} years" if exp >= 2 else
            f"Early-career: {exp} year(s) experience"
        )
    if profile.get('education'):
        strengths.append("Has relevant educational qualifications")
    if profile.get('job_titles'):
        strengths.append(f"Previous role: {profile['job_titles'][0]}")
    strengths.append(
        "Exceptional alignment with job requirements" if score >= 85 else
        "Strong match with most key requirements"     if score >= 70 else
        "Moderate alignment with core requirements"   if score >= 55 else
        "Some relevant qualifications present"
    )

    weaknesses = []
    if missing:
        weaknesses.append(f"Missing JD-required skills: {', '.join(s.title() for s in missing[:6])}")
    if exp is None:
        weaknesses.append("No clear experience information found in CV")
    elif exp < 2:
        weaknesses.append("Limited professional experience (under 2 years)")
    if not profile.get('education'):
        weaknesses.append("No education details found in CV")
    if score < 55:
        weaknesses.append("Significant gaps compared to job requirements")

    return {
        'strengths':       strengths[:5],
        'weaknesses':      weaknesses[:4],
        'skills_matched':  [s.title() for s in matched],
        'skills_missing':  [s.title() for s in missing],
        'skills_extra':    [s.title() for s in extra[:8]],
        'llm_narrative':   '',
        'llm_highlights':  [],
        'llm_concerns':    [],
        'interview_focus': [],
    }

# =============================================================================
# GEMINI ON-DEMAND EXPLANATION
# =============================================================================
def gemini_explain(job_desc, cv_text, profile, score, base_explanation):
    if not GEMINI_AVAILABLE:
        return base_explanation

    skills_str = ', '.join(profile.get('skills', [])[:20]) or 'Not listed'
    edu_str    = ' | '.join(profile.get('education', [])[:2]) or 'Not listed'
    roles_str  = ' | '.join(profile.get('job_titles', [])[:3]) or 'Not listed'
    exp        = profile.get('experience_years')
    companies  = ', '.join(profile.get('companies', [])[:3]) or 'Not listed'

    prompt = f"""You are a senior technical recruiter with 15 years of experience.
Evaluate this candidate for the role below. Return ONLY valid JSON — no markdown, no explanation.

JOB DESCRIPTION:
---
{job_desc[:2000]}
---

CANDIDATE:
- Name: {profile.get('name', 'Unknown')}
- Experience: {f"{exp} years" if exp else "Not specified"}
- Skills: {skills_str}
- Education: {edu_str}
- Previous Roles: {roles_str}
- Companies: {companies}
- Match Score: {score}/100

CV EXCERPT:
---
{cv_text[:1200]}
---

Return exactly this JSON:
{{
  "recruiter_narrative": "3-4 sentences assessing fit for THIS specific role. Be specific.",
  "key_highlights": ["highlight 1", "highlight 2", "highlight 3"],
  "concerns": ["concern 1", "concern 2"],
  "interview_topics": ["topic 1", "topic 2", "topic 3"]
}}"""

    try:
        response = GEMINI_CLIENT.generate_content(prompt)
        raw      = response.text.strip()
        raw      = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw      = re.sub(r'\s*```\s*$',       '', raw, flags=re.MULTILINE)
        result   = json.loads(raw.strip())
        if isinstance(result, dict):
            base_explanation['llm_narrative']   = result.get('recruiter_narrative', '')
            base_explanation['llm_highlights']  = result.get('key_highlights', [])
            base_explanation['llm_concerns']    = result.get('concerns', [])
            base_explanation['interview_focus'] = result.get('interview_topics', [])
            print(f"[GEMINI] Enhanced explanation for {profile.get('name','?')}")
    except json.JSONDecodeError as e:
        print(f"[GEMINI] JSON parse error: {e}")
    except Exception as e:
        print(f"[GEMINI] API error: {e}")

    return base_explanation

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rankcv-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH']             = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI']        = 'mysql+pymysql://root:root@localhost/rankcv_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# =============================================================================
# EMAIL
# =============================================================================
EMAIL_CONFIG = {
    'SMTP_SERVER':     'smtp.gmail.com',
    'SMTP_PORT':       587,
    'SENDER_EMAIL':    'francisonyedikachiagwu@gmail.com',
    'SENDER_PASSWORD': os.environ.get('EMAIL_PASSWORD', ''),
    'RECIPIENT_EMAIL': 'francisonyedikachiagwu@gmail.com',
}

def send_contact_email(name, email, subject, message):
    try:
        msg             = MIMEMultipart('alternative')
        msg['Subject']  = f"[RankCV Contact] {subject}"
        msg['From']     = EMAIL_CONFIG['SENDER_EMAIL']
        msg['To']       = EMAIL_CONFIG['RECIPIENT_EMAIL']
        msg['Reply-To'] = email
        msg.attach(MIMEText(f"From: {name} <{email}>\n\n{message}", 'plain'))
        with smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT']) as s:
            s.starttls()
            s.login(EMAIL_CONFIG['SENDER_EMAIL'], EMAIL_CONFIG['SENDER_PASSWORD'])
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"[EMAIL] {e}")
        return False

def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

@app.context_processor
def inject_now():
    return {'now': utc_now()}

# =============================================================================
# PDF TEXT EXTRACTION
# =============================================================================
def extract_text_from_pdf(file):
    try:
        file.seek(0)
        content = file.read()
        if not content:
            return None, "File is empty"
        buf  = io.BytesIO(content)
        text = ""
        if HAS_PDFPLUMBER:
            with pdfplumber.open(buf) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
        else:
            reader = PyPDF2.PdfReader(buf)
            for page in reader.pages:
                try:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
                except Exception:
                    continue
        clean = text.strip()
        return (clean, None) if clean else (None, "No text found — may be a scanned PDF")
    except Exception as e:
        return None, str(e)

# =============================================================================
# DATABASE MODELS
# =============================================================================
class User(db.Model):
    __tablename__        = 'users'
    id                   = db.Column(db.Integer, primary_key=True)
    email                = db.Column(db.String(120), unique=True, nullable=False)
    password_hash        = db.Column(db.String(255))
    profile_pic          = db.Column(db.String(255), nullable=True)
    subscription_plan    = db.Column(db.String(20), default='free')
    subscription_expires = db.Column(db.DateTime, nullable=True)
    stripe_customer_id   = db.Column(db.String(255), nullable=True)
    created_at           = db.Column(db.TIMESTAMP, default=utc_now)
    updated_at           = db.Column(db.TIMESTAMP, default=utc_now, onupdate=utc_now)
    usage_logs = db.relationship('UsageLog',       backref='user', lazy=True, cascade='all, delete-orphan')
    settings   = db.relationship('UserSettings',   backref='user', uselist=False, cascade='all, delete-orphan')
    history    = db.relationship('RankingHistory',  backref='user', lazy=True, cascade='all, delete-orphan')

class UsageLog(db.Model):
    __tablename__  = 'usage_logs'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_date   = db.Column(db.Date, nullable=False)
    sessions_used  = db.Column(db.Integer, default=0)
    cvs_processed  = db.Column(db.Integer, default=0)
    created_at     = db.Column(db.TIMESTAMP, default=utc_now)
    __table_args__ = (db.UniqueConstraint('user_id', 'session_date', name='unique_user_date'),)

class UserSettings(db.Model):
    __tablename__       = 'user_settings'
    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    experience_enabled  = db.Column(db.Boolean, default=True)
    min_experience      = db.Column(db.Integer, default=0)
    min_experience_unit = db.Column(db.String(10), default='years')
    max_experience      = db.Column(db.Integer, default=10)
    max_experience_unit = db.Column(db.String(10), default='years')
    show_score_bar      = db.Column(db.Boolean, default=True)
    save_cvs            = db.Column(db.Boolean, default=False)
    cv_limit            = db.Column(db.Integer, default=20)

class RankingHistory(db.Model):
    __tablename__ = 'ranking_history'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    job_title     = db.Column(db.String(255), nullable=False, default='Job Ranking')
    date_created  = db.Column(db.TIMESTAMP, default=utc_now)
    cv_count      = db.Column(db.Integer, default=0)
    top_score     = db.Column(db.DECIMAL(5, 2), default=0)
    top_candidate = db.Column(db.String(255))
    results_json  = db.Column(db.Text)

# =============================================================================
# AUTH
# =============================================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

PLAN_LIMITS = {
    'free':    {'cvs_per_session': 10,           'max_sessions': 1,            'price': 0},
    'basic':   {'cvs_per_session': 25,           'max_sessions': 2,            'price': 10},
    'pro':     {'cvs_per_session': 50,           'max_sessions': 3,            'price': 20},
    'premium': {'cvs_per_session': float('inf'), 'max_sessions': float('inf'), 'price': 50},
}

def get_current_user():
    if 'user_id' not in session:
        return None
    return db.session.get(User, session['user_id'])

def check_subscription_status(user):
    if not user:
        return False
    if user.subscription_plan == 'free':
        return True
    if user.subscription_expires and user.subscription_expires < utc_now():
        user.subscription_plan = 'free'
        db.session.commit()
        return False
    return True

def check_usage_limit(user_id, cvs_count):
    user = db.session.get(User, user_id)
    if not user:
        return False, "User not found", {}
    if not check_subscription_status(user):
        return False, "Your subscription has expired.", {'limit_type': 'subscription'}
    today  = date.today()
    plan   = user.subscription_plan
    limits = PLAN_LIMITS[plan]
    usage  = UsageLog.query.filter_by(user_id=user_id, session_date=today).first()
    if not usage:
        usage = UsageLog(user_id=user_id, session_date=today, sessions_used=0, cvs_processed=0)
        db.session.add(usage)
        db.session.commit()
    if usage.sessions_used >= limits['max_sessions']:
        return False, (
            f"Daily session limit reached! {usage.sessions_used}/{int(limits['max_sessions'])} sessions used today."
        ), {'limit_type': 'sessions'}
    if cvs_count > limits['cvs_per_session']:
        return False, (
            f"CV limit exceeded! {plan.capitalize()} plan allows {int(limits['cvs_per_session'])} CVs per session."
        ), {'limit_type': 'cvs'}
    return True, "OK", {'sessions_used': usage.sessions_used, 'plan': plan}

def record_usage(user_id, cvs_count):
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user_id, session_date=today).first()
    if not usage:
        usage = UsageLog(user_id=user_id, session_date=today, sessions_used=0, cvs_processed=0)
        db.session.add(usage)
        db.session.commit()
    usage.sessions_used += 1
    usage.cvs_processed += cvs_count
    db.session.commit()

def save_to_history(user_id, job_title, cv_count, results):
    if not results:
        return
    storage = []
    for r in results:
        storage.append({
            'name':             r.get('name', 'Unknown'),
            'filename':         r.get('filename', ''),
            'score':            r.get('score', 0),
            'summary':          r.get('summary', ''),
            'experience_years': r.get('experience_years'),
            'education':        r.get('education', []),
            'previous_roles':   r.get('previous_roles', []),
            'companies':        r.get('companies', []),
            'strengths':        r.get('strengths', []),
            'weaknesses':       r.get('weaknesses', []),
            'skills_matched':   r.get('skills_matched', []),
            'skills_missing':   r.get('skills_missing', []),
            'skills_extra':     r.get('skills_extra', []),
            'llm_narrative':    r.get('llm_narrative', ''),
            'llm_highlights':   r.get('llm_highlights', []),
            'llm_concerns':     r.get('llm_concerns', []),
            'interview_focus':  r.get('interview_focus', []),
            'profile':          r.get('profile', {}),
        })
    top = results[0] if results else None
    db.session.add(RankingHistory(
        user_id=user_id,
        job_title=(job_title[:50] if job_title else 'Untitled'),
        cv_count=cv_count,
        top_score=top['score'] if top else 0,
        top_candidate=(top.get('name', 'Unknown')[:50] if top else 'Unknown'),
        results_json=json.dumps(storage),
    ))
    db.session.commit()

# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        valid    = False
        if user and user.password_hash:
            if check_password_hash(user.password_hash, password):
                valid = True
            elif user.password_hash == hashlib.sha256(password.encode()).hexdigest():
                user.password_hash = generate_password_hash(password)
                db.session.commit()
                valid = True
        if user and valid:
            session['user_id']          = user.id
            session['user_email']       = user.email
            session['user_profile_pic'] = user.profile_pic or ''
            flash(f'Welcome back, {email}!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            flash('Invalid email address.', 'error')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        user = User(email=email, password_hash=generate_password_hash(password), subscription_plan='free')
        db.session.add(user)
        db.session.commit()
        db.session.add(UserSettings(user_id=user.id))
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST' and 'profile_pic' in request.files:
        file = request.files['profile_pic']
        if file and file.filename:
            filename   = secure_filename(f"user_{user.id}_{file.filename}")
            if not allowed_file(filename):
                flash('Invalid file type.', 'error')
                return redirect(url_for('profile'))
            upload_dir = os.path.join(app.static_folder, 'uploads', 'profile_pics')
            os.makedirs(upload_dir, exist_ok=True)
            file.save(os.path.join(upload_dir, filename))
            if user.profile_pic:
                old = os.path.join(app.static_folder, user.profile_pic)
                if os.path.exists(old):
                    os.remove(old)
            user.profile_pic = f"uploads/profile_pics/{filename}"
            db.session.commit()
            session['user_profile_pic'] = user.profile_pic
            flash('Profile picture updated!', 'success')
            return redirect(url_for('profile'))
    today       = date.today()
    usage       = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    plan_limits = PLAN_LIMITS[user.subscription_plan]
    return render_template('profile.html', user=user, plan_limits=plan_limits,
                           sessions_used=usage.sessions_used if usage else 0)

# =============================================================================
# RANK ROUTE
# =============================================================================
@app.route('/rank', methods=['POST'])
@login_required
def rank():
    user = get_current_user()
    if not user:
        flash('Session expired.', 'error')
        return redirect(url_for('login'))

    job_description = request.form.get('job_description', '').strip()
    files           = request.files.getlist('cvs')
    valid_files     = [f for f in files if f.filename != '']

    allowed, message, _ = check_usage_limit(user.id, len(valid_files))
    if not allowed:
        flash(message, 'error')
        return redirect('/')
    if not job_description:
        flash('Please enter a job description.', 'warning')
        return redirect('/')
    if not valid_files:
        flash('Please upload at least one CV.', 'warning')
        return redirect('/')

    # Step 1: Extract PDF text
    cv_data, failed_files = [], []
    for f in valid_files:
        text, err = extract_text_from_pdf(f)
        if text:
            cv_data.append({'filename': f.filename, 'text': text})
        else:
            failed_files.append(f"{f.filename}: {err}")

    if failed_files:
        flash("Some files could not be read:<br>" + "<br>".join(failed_files), 'warning')
    if not cv_data:
        flash('No readable CVs found. Please upload text-based PDFs.', 'error')
        return redirect('/')

    # Step 2: Parse CVs with regex (instant — name, skills, education)
    print(f"\n{'='*55}\nRANKING: {len(cv_data)} CVs\n{'='*55}")
    parsed = []
    for cv in cv_data:
        parser  = CVParser(cv['text'], cv['filename'])
        profile = parser.get_full_profile()
        parsed.append({'filename': cv['filename'], 'text': cv['text'], 'profile': profile})
        print(f"  ✓ {profile['name']} | Skills: {len(profile['skills'])} | Exp: {profile['experience_years']}yr")

    # Step 3: Semantic scoring (sentence-transformers primary, TF-IDF fallback)
    cv_texts = [p['text']    for p in parsed]
    profiles = [p['profile'] for p in parsed]
    scores   = compute_scores(job_description, cv_texts, profiles)

    # Step 4: Rule-based explanations (instant)
    results = []
    for cv_info, score in zip(parsed, scores):
        profile = cv_info['profile']
        exp     = rule_based_explanation(job_description, profile, score)
        results.append({
            'name':             profile['name'],   # ← CV content name, NOT filename
            'filename':         cv_info['filename'],
            'score':            score,
            'summary':          profile.get('summary', ''),
            'experience_years': profile.get('experience_years'),
            'education':        profile.get('education', []),
            'previous_roles':   profile.get('job_titles', []),
            'companies':        profile.get('companies', []),
            'strengths':        exp['strengths'],
            'weaknesses':       exp['weaknesses'],
            'skills_matched':   exp['skills_matched'],
            'skills_missing':   exp['skills_missing'],
            'skills_extra':     exp['skills_extra'],
            'llm_narrative':    '',
            'llm_highlights':   [],
            'llm_concerns':     [],
            'interview_focus':  [],
            'profile':          profile,
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Final ranking:")
    for i, r in enumerate(results, 1):
        print(f"    #{i} {r['name']} — {r['score']}%")

    record_usage(user.id, len(valid_files))
    save_to_history(user.id, job_description[:100], len(valid_files), results)

    return render_template('new_ranking.html', job=job_description, results=results, view_mode=False)

# =============================================================================
# GEMINI ON-DEMAND ENDPOINT
# =============================================================================
@app.route('/api/enhance-explanation', methods=['POST'])
@login_required
def enhance_explanation():
    data     = request.json or {}
    cv_text  = data.get('cv_text', '')
    job_desc = data.get('job_desc', '')
    profile  = data.get('profile', {})
    score    = data.get('score', 0)
    base_exp = data.get('base_explanation', {})

    if not GEMINI_AVAILABLE:
        return jsonify({'success': False, 'reason': 'Gemini not configured'})

    enhanced = gemini_explain(job_desc, cv_text, profile, score, base_exp)
    return jsonify({
        'success':         True,
        'llm_narrative':   enhanced.get('llm_narrative', ''),
        'llm_highlights':  enhanced.get('llm_highlights', []),
        'llm_concerns':    enhanced.get('llm_concerns', []),
        'interview_focus': enhanced.get('interview_focus', []),
    })

# =============================================================================
# HISTORY
# =============================================================================
@app.route('/history')
@login_required
def history():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    rows = RankingHistory.query.filter_by(user_id=user.id).order_by(RankingHistory.date_created.desc()).all()
    history_data = [{
        'id':            h.id,
        'job_title':     h.job_title,
        'date':          h.date_created.strftime('%b %d, %Y'),
        'date_raw':      h.date_created.isoformat(),
        'cv_count':      h.cv_count,
        'top_score':     float(h.top_score),
        'top_candidate': h.top_candidate,
    } for h in rows]
    return render_template('history.html', history_data=history_data)

@app.route('/history/view/<int:history_id>')
@login_required
def view_history_detail(history_id):
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    entry = RankingHistory.query.filter_by(id=history_id, user_id=user.id).first_or_404()
    try:
        results = json.loads(entry.results_json) if entry.results_json else []
    except Exception:
        results = []
    return render_template('new_ranking.html', job=entry.job_title,
                           results=results, view_mode=True, history_id=history_id)

# =============================================================================
# SETTINGS
# =============================================================================
@app.route('/settings')
@login_required
def settings():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    today         = date.today()
    usage         = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    user_settings = UserSettings.query.filter_by(user_id=user.id).first()
    return render_template('settings.html', user=user,
                           user_settings=user_settings,
                           plan_limits=PLAN_LIMITS[user.subscription_plan],
                           sessions_used=usage.sessions_used if usage else 0,
                           PLAN_LIMITS=PLAN_LIMITS)

@app.route('/api/settings', methods=['POST'])
@login_required
def save_settings():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    s    = UserSettings.query.filter_by(user_id=user.id).first()
    if not s:
        s = UserSettings(user_id=user.id)
        db.session.add(s)
    s.experience_enabled  = data.get('experienceEnabled', True)
    s.min_experience      = data.get('minExperience', 0)
    s.min_experience_unit = data.get('minExperienceUnit', 'years')
    s.max_experience      = data.get('maxExperience', 10)
    s.max_experience_unit = data.get('maxExperienceUnit', 'years')
    s.show_score_bar      = data.get('showScoreBar', True)
    s.save_cvs            = data.get('saveCvs', False)
    s.cv_limit            = data.get('cvLimit', 20)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/usage-status')
@login_required
def usage_status():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    plan  = PLAN_LIMITS[user.subscription_plan]
    return jsonify({
        'plan':                 user.subscription_plan,
        'sessions_used':        usage.sessions_used if usage else 0,
        'sessions_limit':       plan['max_sessions'] if plan['max_sessions'] != float('inf') else 'Unlimited',
        'cvs_per_session':      plan['cvs_per_session'] if plan['cvs_per_session'] != float('inf') else 'Unlimited',
        'subscription_expires': user.subscription_expires.isoformat() if user.subscription_expires else None,
    })

@app.route('/subscribe/<plan>')
@login_required
def subscribe(plan):
    if plan not in PLAN_LIMITS:
        return "Invalid plan", 400
    if request.args.get('token') != 'demo-confirm':
        flash('Invalid subscription request.', 'error')
        return redirect(url_for('pricing'))
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    user.subscription_plan    = plan
    user.subscription_expires = datetime(2026, 12, 31)
    db.session.commit()
    flash(f'Upgraded to {plan.capitalize()} plan!', 'success')
    return redirect(url_for('settings'))

@app.route('/api/delete-history', methods=['POST'])
@login_required
def delete_history():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    RankingHistory.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return jsonify({'success': True})

# =============================================================================
# STATIC PAGES
# =============================================================================
@app.route('/pricing')
@login_required
def pricing():
    user = get_current_user()
    return render_template('pricing.html', user=user, PLAN_LIMITS=PLAN_LIMITS,
                           current_plan=user.subscription_plan)

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/changelog')
def changelog():
    return render_template('changelog.html')

@app.route('/presentation')
def presentation():
    return app.send_static_file('rankcv_present.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        email   = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        if not name or not email or not message:
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('contact'))
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            flash('Invalid email address.', 'error')
            return redirect(url_for('contact'))
        send_contact_email(name, email, subject or 'General Inquiry', message)
        flash(f'Thank you {name}! We will respond within 24 hours.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.errorhandler(404)
def not_found(e):
    return ("<h1 style='text-align:center;margin-top:50px'>404 — Not Found</h1>"
            "<p style='text-align:center'><a href='/'>Go Home</a></p>"), 404

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum is 16MB.', 'error')
    return redirect('/')

# =============================================================================
# STARTUP
# =============================================================================
if __name__ == '__main__':
    import os
    with app.app_context():
        db.create_all()
        print("\n" + "="*55)
        print(" RankCv - Ready")
        print("-"*55)
        print(f" Scoring  : {'Semantic embeddings (sentence-transformers)' if HAS_EMBEDDINGS else 'TF-IDF fallback'}")
        print(f" Gemini   : {'ON - AI explanations on-demand' if GEMINI_AVAILABLE else 'OFF - rule-based fallback'}")
        print(f" Database : MySQL( rankcv_db )")
        print("-"*55)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
