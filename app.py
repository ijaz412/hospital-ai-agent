import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from groq import Groq

# =========================================================
# AI HOSPITAL COMPLAINT AGENT - CONVERSATIONAL MVP
# =========================================================

DB_PATH = "hospital_complaints.db"
GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="AI Hospital Complaint Agent",
    page_icon="🏥",
    layout="wide",
)


# =========================================================
# MODERN UI / VISUAL DESIGN
# =========================================================
st.markdown("""
<style>
    /* App background */
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(0, 200, 180, 0.10), transparent 25%),
            radial-gradient(circle at 10% 20%, rgba(50, 110, 255, 0.08), transparent 28%),
            #0b1020;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d1322 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
    }

    /* Hero */
    .hero {
        padding: 28px 30px;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(31, 41, 55, 0.95), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 18px 50px rgba(0,0,0,0.28);
        margin-bottom: 20px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(20, 184, 166, 0.14);
        border: 1px solid rgba(20, 184, 166, 0.25);
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        line-height: 1.1;
        letter-spacing: -1px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.02rem;
        margin: 10px 0 0 0;
    }

    /* Feature cards */
    .feature-card {
        min-height: 145px;
        padding: 20px;
        border-radius: 18px;
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
        transition: transform 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }

    .feature-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 6px;
    }

    .feature-text {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        margin-bottom: 10px;
    }

    /* Inputs */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 650;
        border: 1px solid rgba(255,255,255,0.10);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    }

    /* Status pills */
    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(59,130,246,0.14);
        border: 1px solid rgba(59,130,246,0.25);
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Small section heading */
    .section-kicker {
        color: #5eead4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.72rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        padding: 28px 0 5px;
    }

    /* Hide Streamlit branding/footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

CATEGORIES = [
    "Long Waiting Time",
    "Doctor/Staff Availability",
    "Staff Behavior/Communication",
    "Medicine Problem",
    "Laboratory/Test Problem",
    "Radiology/Imaging Problem",
    "OT/Surgery Service Problem",
    "Cleanliness/Hygiene",
    "Equipment Problem",
    "Medical Records Problem",
    "Blood Bank Problem",
    "Security/Safety",
    "Ambulance/Transport",
    "Billing/Accounts",
    "General Administration",
]

DEPARTMENTS = [
    "OPD",
    "Emergency",
    "IPD/Ward",
    "Nursing",
    "Pharmacy",
    "Laboratory",
    "Radiology",
    "Operation Theatre",
    "Sanitation",
    "Blood Bank",
    "Medical Records",
    "Security",
    "Ambulance/Transport",
    "Accounts/Billing",
    "Administration",
]

PRIORITIES = ["Low", "Medium", "High", "Immediate Attention"]

STATUSES = [
    "Submitted",
    "Assigned",
    "Under Review",
    "Action Taken",
    "Escalated",
    "Resolved",
    "Closed",
]


# =========================================================
# DATABASE
# =========================================================

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    con = get_db()

    con.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            patient_name TEXT,
            patient_contact TEXT,
            hospital TEXT NOT NULL,
            location TEXT,
            original_text TEXT NOT NULL,
            language TEXT,
            category TEXT,
            department TEXT,
            priority TEXT,
            summary TEXT,
            status TEXT DEFAULT 'Submitted',
            assigned_to TEXT,
            staff_note TEXT,
            resolution TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            due_at TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            updated_by TEXT,
            created_at TEXT NOT NULL
        )
    """)

    con.commit()
    con.close()


init_db()


# =========================================================
# GROQ AI
# =========================================================

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def analyze_conversation(messages):
    """
    Analyze the whole patient conversation and decide:
    1. Whether more information is required.
    2. What the next question should be.
    3. Whether the complaint is ready for confirmation.
    4. Structured complaint information.
    """

    client = get_groq_client()

    if client is None:
        return {
            "ready": True,
            "next_question": "",
            "language": "Unknown",
            "category": "General Administration",
            "department": "Administration",
            "priority": "Medium",
            "summary": "AI API key is not configured.",
            "missing_information": [],
            "safety_flag": False,
            "safety_message": "",
        }

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
    )

    system_prompt = f"""
You are the conversational AI agent for a hospital patient grievance system in Pakistan.

Your job is to help a patient report a hospital service complaint.

The patient may use:
- English
- Urdu
- Roman Urdu
- Mixed English/Urdu/Roman Urdu

IMPORTANT:
You are NOT a doctor.
You must NOT diagnose diseases.
You must NOT prescribe medicines.
You must NOT make legal conclusions.
You must NOT declare negligence as a fact.
You must NOT replace hospital staff.
You must NOT delay emergency care.

Your workflow:

STEP 1:
Understand what the patient is saying.

STEP 2:
Ask only ONE useful follow-up question at a time if important information is missing.

STEP 3:
When enough information is available, set ready=true.

Important information may include:
- hospital name
- department/ward/location
- what happened
- approximate time/date
- basic service problem

Do NOT ask for unnecessary sensitive medical information.

If the patient already provided enough information, DO NOT ask unnecessary questions.

If the patient describes an immediate medical danger:
- safety_flag=true
- priority="Immediate Attention"
- provide a short safety_message telling them to immediately alert emergency medical staff.
- The complaint process must not delay urgent medical care.

Allowed categories:
{", ".join(CATEGORIES)}

Allowed departments:
{", ".join(DEPARTMENTS)}

Allowed priorities:
{", ".join(PRIORITIES)}

Return ONLY valid JSON with exactly these keys:

{{
  "ready": true,
  "next_question": "",
  "language": "English|Urdu|Roman Urdu|Mixed",
  "category": "one allowed category",
  "department": "one allowed department",
  "priority": "Low|Medium|High|Immediate Attention",
  "summary": "short professional English summary",
  "missing_information": ["important missing item"],
  "safety_flag": false,
  "safety_message": ""
}}

If more information is needed:
- ready=false
- next_question must contain exactly ONE question.
- missing_information should identify what is missing.

If ready=true:
- next_question should be empty.
- summary should be complete enough for hospital staff.

The AI priority is only a recommendation. A hospital officer must make the final decision.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text},
            ],
        )

        result = json.loads(response.choices[0].message.content)

        # Safety/fallback validation
        if result.get("category") not in CATEGORIES:
            result["category"] = "General Administration"

        if result.get("department") not in DEPARTMENTS:
            result["department"] = "Administration"

        if result.get("priority") not in PRIORITIES:
            result["priority"] = "Medium"

        if result.get("language") not in [
            "English", "Urdu", "Roman Urdu", "Mixed"
        ]:
            result["language"] = "Unknown"

        if not isinstance(result.get("missing_information"), list):
            result["missing_information"] = []

        result["ready"] = bool(result.get("ready", False))

        return result

    except Exception as e:
        return {
            "ready": False,
            "next_question": "Could you please provide a little more information about what happened?",
            "language": "Unknown",
            "category": "General Administration",
            "department": "Administration",
            "priority": "Medium",
            "summary": "",
            "missing_information": ["complaint details"],
            "safety_flag": False,
            "safety_message": "",
            "error": str(e),
        }


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def create_complaint_id():
    return (
        "HCA-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )


def save_complaint(data):
    cid = create_complaint_id()
    now = datetime.now()

    # Demo SLA values. Change according to the hospital's actual policy.
    sla_hours = {
        "Immediate Attention": 2,
        "High": 6,
        "Medium": 24,
        "Low": 72,
    }

    due = now + timedelta(
        hours=sla_hours.get(data["priority"], 24)
    )

    con = get_db()

    con.execute("""
        INSERT INTO complaints (
            complaint_id,
            patient_name,
            patient_contact,
            hospital,
            location,
            original_text,
            language,
            category,
            department,
            priority,
            summary,
            status,
            created_at,
            updated_at,
            due_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Submitted', ?, ?, ?)
    """, (
        cid,
        data.get("patient_name", ""),
        data.get("patient_contact", ""),
        data["hospital"],
        data.get("location", ""),
        data["original_text"],
        data.get("language", "Unknown"),
        data.get("category", "General Administration"),
        data.get("department", "Administration"),
        data.get("priority", "Medium"),
        data.get("summary", ""),
        now.isoformat(timespec="seconds"),
        now.isoformat(timespec="seconds"),
        due.isoformat(timespec="seconds"),
    ))

    con.execute("""
        INSERT INTO updates (
            complaint_id,
            status,
            note,
            updated_by,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        cid,
        "Submitted",
        "Complaint submitted by patient.",
        "Patient",
        now.isoformat(timespec="seconds"),
    ))

    con.commit()
    con.close()

    return cid


def get_all_complaints():
    con = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM complaints ORDER BY id DESC",
        con
    )
    con.close()
    return df


def get_complaint(cid):
    con = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM complaints WHERE complaint_id=?",
        con,
        params=(cid.strip().upper(),)
    )
    con.close()
    return df


def get_updates(cid):
    con = get_db()

    df = pd.read_sql_query(
        """
        SELECT status, note, updated_by, created_at
        FROM updates
        WHERE complaint_id=?
        ORDER BY id
        """,
        con,
        params=(cid,)
    )

    con.close()
    return df


def update_complaint(
    cid,
    status,
    assigned_to,
    note,
    resolution
):
    now = datetime.now().isoformat(timespec="seconds")

    con = get_db()

    con.execute("""
        UPDATE complaints
        SET status=?,
            assigned_to=?,
            staff_note=?,
            resolution=?,
            updated_at=?
        WHERE complaint_id=?
    """, (
        status,
        assigned_to,
        note,
        resolution,
        now,
        cid,
    ))

    con.execute("""
        INSERT INTO updates (
            complaint_id,
            status,
            note,
            updated_by,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        cid,
        status,
        note,
        assigned_to or "Hospital Staff",
        now,
    ))

    con.commit()
    con.close()


# =========================================================
# PATIENT CHAT
# =========================================================

def reset_patient_chat():
    st.session_state.patient_messages = []
    st.session_state.patient_started = False
    st.session_state.patient_analysis = None
    st.session_state.patient_data = None
    st.session_state.patient_ready = False
    st.session_state.patient_submitted_id = None


def add_patient_message(role, content):
    st.session_state.patient_messages.append({
        "role": role,
        "content": content,
    })


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="padding: 8px 2px 18px;">
    <div style="font-size:2.2rem;">🏥</div>
    <div style="font-size:1.35rem;font-weight:800;line-height:1.15;">
        AI Hospital<br>Complaint Agent
    </div>
    <div style="color:#94a3b8;font-size:.78rem;margin-top:8px;">
        Listen • Understand • Route • Track
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Menu",
    [
        "Patient - AI Complaint",
        "Patient - Track Complaint",
        "Hospital Staff Dashboard",
        "Analytics",
        "About",
    ]
)

st.sidebar.info(
    "This is an MVP. AI assists the complaint process. "
    "Authorized hospital staff make final decisions."
)


# =========================================================
# PATIENT CONVERSATIONAL AGENT
# =========================================================

if page == "Patient - AI Complaint":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🤖 AI-ASSISTED • MULTILINGUAL • PATIENT-FIRST</div>
        <div class="hero h1"></div>
        <h1>🏥 Your Voice Matters</h1>
        <p>
            Tell us what happened in your own words. Our AI helps turn your
            message into a clear hospital complaint and guides it to the
            right department.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">How it works</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">1. Tell your story</div>
            <div class="feature-text">Write naturally in English, Urdu, Roman Urdu, or mixed language.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">2. AI understands</div>
            <div class="feature-text">The agent asks only useful follow-up questions and structures the complaint.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">3. Route & track</div>
            <div class="feature-text">A complaint ID is created so you can follow its progress.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.caption("💡 You can describe the problem just as you would explain it to a hospital staff member.")

    st.warning(
        "🚨 If someone needs immediate medical help, alert the "
        "hospital's emergency medical staff immediately. "
        "Do not wait for the complaint process."
    )

    if "patient_messages" not in st.session_state:
        reset_patient_chat()

    if not st.session_state.patient_started:

        st.markdown("""
        <div class="feature-card" style="margin: 18px 0;">
            <div class="feature-icon">✨</div>
            <div class="feature-title">Try an example</div>
            <div class="feature-text">
                “Meri ammi emergency mein hain aur 2 ghantay se doctor nahi aya.”
                <br><br>
                The AI can understand this and ask for the missing details step by step.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("start_complaint_form"):
            patient_name = st.text_input(
                "Patient/Complainant Name (optional)"
            )

            patient_contact = st.text_input(
                "Phone/Contact (optional)"
            )

            hospital = st.text_input(
                "Hospital Name *"
            )

            location = st.text_input(
                "Department/Ward/Location (optional)"
            )

            first_message = st.text_area(
                "What happened? *",
                height=160,
                placeholder=(
                    "Meri ammi emergency mein hain aur "
                    "2 ghantay se doctor nahi aya."
                )
            )

            start = st.form_submit_button(
                "🤖 Start AI Complaint",
                use_container_width=True
            )

        if start:

            if not hospital.strip():
                st.error("Please enter the hospital name.")

            elif not first_message.strip():
                st.error("Please describe what happened.")

            else:

                st.session_state.patient_data = {
                    "patient_name": patient_name.strip(),
                    "patient_contact": patient_contact.strip(),
                    "hospital": hospital.strip(),
                    "location": location.strip(),
                }

                st.session_state.patient_started = True

                add_patient_message(
                    "user",
                    first_message.strip()
                )

                with st.spinner("AI is understanding your complaint..."):
                    result = analyze_conversation(
                        st.session_state.patient_messages
                    )

                st.session_state.patient_analysis = result

                if result.get("ready"):
                    st.session_state.patient_ready = True
                else:
                    if result.get("next_question"):
                        add_patient_message(
                            "assistant",
                            result["next_question"]
                        )

                st.rerun()

    else:

        st.markdown(
            '<span class="status-pill">🟢 AI complaint conversation active</span>',
            unsafe_allow_html=True
        )
        st.write("")

        # Display conversation
        for message in st.session_state.patient_messages:

            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])

            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

        # If complaint is not ready, continue conversation
        if not st.session_state.patient_ready:

            user_answer = st.chat_input(
                "Type your answer..."
            )

            if user_answer:

                add_patient_message(
                    "user",
                    user_answer
                )

                with st.spinner("AI is processing your answer..."):
                    result = analyze_conversation(
                        st.session_state.patient_messages
                    )

                st.session_state.patient_analysis = result

                if result.get("safety_flag"):
                    st.warning(
                        "⚠️ " + result.get(
                            "safety_message",
                            "Please alert emergency medical staff immediately."
                        )
                    )

                if result.get("ready"):

                    st.session_state.patient_ready = True

                else:

                    question = result.get("next_question")

                    if question:
                        add_patient_message(
                            "assistant",
                            question
                        )

                st.rerun()

        # Ready for confirmation
        if st.session_state.patient_ready:

            result = st.session_state.patient_analysis
            patient_data = st.session_state.patient_data

            st.divider()

            st.subheader("📋 Complaint Summary")

            st.write(
                "The AI has collected enough information. "
                "Please review everything before submission."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Category",
                result.get("category", "General Administration")
            )

            c2.metric(
                "Department",
                result.get("department", "Administration")
            )

            c3.metric(
                "AI Priority",
                result.get("priority", "Medium")
            )

            st.write(
                "**Detected language:**",
                result.get("language", "Unknown")
            )

            if result.get("safety_flag"):
                st.warning(
                    "⚠️ " + result.get(
                        "safety_message",
                        "Please alert emergency medical staff immediately."
                    )
                )

            # Editable confirmation fields
            category = st.selectbox(
                "Category",
                CATEGORIES,
                index=(
                    CATEGORIES.index(result["category"])
                    if result.get("category") in CATEGORIES
                    else len(CATEGORIES) - 1
                )
            )

            department = st.selectbox(
                "Department",
                DEPARTMENTS,
                index=(
                    DEPARTMENTS.index(result["department"])
                    if result.get("department") in DEPARTMENTS
                    else len(DEPARTMENTS) - 1
                )
            )

            priority = st.selectbox(
                "Priority",
                PRIORITIES,
                index=(
                    PRIORITIES.index(result["priority"])
                    if result.get("priority") in PRIORITIES
                    else 1
                )
            )

            summary = st.text_area(
                "Complaint Summary",
                value=result.get("summary", ""),
                height=130
            )

            st.subheader("Original Conversation")

            conversation_preview = "\n\n".join(
                f"{'Patient' if m['role']=='user' else 'AI'}: {m['content']}"
                for m in st.session_state.patient_messages
            )

            st.text_area(
                "Conversation",
                conversation_preview,
                height=200,
                disabled=True
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "✏️ Continue / Add More Information",
                    use_container_width=True
                ):
                    st.session_state.patient_ready = False

                    add_patient_message(
                        "assistant",
                        "Sure. Please tell me any additional information you want to add."
                    )

                    st.rerun()

            with col2:
                if st.button(
                    "✅ Confirm & Submit Complaint",
                    use_container_width=True
                ):

                    original_text = "\n".join(
                        m["content"]
                        for m in st.session_state.patient_messages
                        if m["role"] == "user"
                    )

                    data = {
                        **patient_data,
                        "original_text": original_text,
                        "language": result.get("language", "Unknown"),
                        "category": category,
                        "department": department,
                        "priority": priority,
                        "summary": summary,
                    }

                    cid = save_complaint(data)

                    st.session_state.patient_submitted_id = cid

            if st.session_state.patient_submitted_id:

                st.success(
                    "✅ Complaint submitted successfully!"
                )

                st.balloons()

                st.markdown(
                    f"## Complaint ID: `{st.session_state.patient_submitted_id}`"
                )

                st.info(
                    "Save this Complaint ID. You can use it in "
                    "**Patient - Track Complaint**."
                )

                if st.button("🆕 Submit Another Complaint"):
                    reset_patient_chat()
                    st.rerun()


# =========================================================
# PATIENT TRACKING
# =========================================================

elif page == "Patient - Track Complaint":

    st.title("🔎 Track Your Complaint")

    cid = st.text_input(
        "Complaint ID",
        placeholder="HCA-20260905-ABC123"
    )

    if st.button(
        "Track Complaint",
        use_container_width=True
    ):

        if not cid.strip():
            st.error("Please enter a Complaint ID.")

        else:

            df = get_complaint(cid)

            if df.empty:

                st.error(
                    "Complaint not found. Please check your ID."
                )

            else:

                row = df.iloc[0]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Status",
                    row["status"]
                )

                c2.metric(
                    "Priority",
                    row["priority"]
                )

                c3.metric(
                    "Department",
                    row["department"]
                )

                st.subheader("Complaint Summary")

                st.write(row["summary"])

                st.subheader("Complaint Timeline")

                timeline = get_updates(
                    row["complaint_id"]
                )

                for _, update in timeline.iterrows():

                    st.write(
                        f"**{update['status']}** — "
                        f"{update['created_at']}"
                    )

                    if update["note"]:
                        st.caption(
                            f"{update['updated_by']}: "
                            f"{update['note']}"
                        )

                if row["resolution"]:

                    st.subheader("Resolution")

                    st.success(
                        row["resolution"]
                    )


# =========================================================
# HOSPITAL STAFF DASHBOARD
# =========================================================

elif page == "Hospital Staff Dashboard":

    st.title("👨‍⚕️ Hospital Staff Dashboard")

    st.caption(
        "Prototype dashboard. Production deployment requires "
        "secure authentication and role-based access."
    )

    df = get_all_complaints()

    if df.empty:

        st.info("No complaints have been submitted yet.")

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total",
            len(df)
        )

        c2.metric(
            "Pending",
            len(
                df[
                    ~df.status.isin(
                        ["Resolved", "Closed"]
                    )
                ]
            )
        )

        c3.metric(
            "High / Immediate",
            len(
                df[
                    df.priority.isin(
                        ["High", "Immediate Attention"]
                    )
                ]
            )
        )

        c4.metric(
            "Resolved",
            len(
                df[
                    df.status.isin(
                        ["Resolved", "Closed"]
                    )
                ]
            )
        )

        st.divider()

        st.subheader("Complaint Queue")

        a, b, c = st.columns(3)

        with a:
            status_filter = st.selectbox(
                "Status",
                ["All"] + sorted(df.status.unique())
            )

        with b:
            department_filter = st.selectbox(
                "Department",
                ["All"] + sorted(df.department.unique())
            )

        with c:
            priority_filter = st.selectbox(
                "Priority",
                ["All"] + sorted(df.priority.unique())
            )

        filtered = df.copy()

        if status_filter != "All":
            filtered = filtered[
                filtered.status == status_filter
            ]

        if department_filter != "All":
            filtered = filtered[
                filtered.department == department_filter
            ]

        if priority_filter != "All":
            filtered = filtered[
                filtered.priority == priority_filter
            ]

        st.dataframe(
            filtered[
                [
                    "complaint_id",
                    "hospital",
                    "category",
                    "department",
                    "priority",
                    "status",
                    "created_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered.empty:

            selected_id = st.selectbox(
                "Review Complaint",
                filtered.complaint_id.tolist()
            )

            row = get_complaint(
                selected_id
            ).iloc[0]

            st.divider()

            st.write(
                "**Original patient conversation:**"
            )

            st.write(
                row["original_text"]
            )

            st.write(
                "**AI Summary:**"
            )

            st.write(
                row["summary"]
            )

            st.write(
                f"**AI Suggested Category:** {row['category']}"
            )

            st.write(
                f"**AI Suggested Department:** {row['department']}"
            )

            st.write(
                f"**AI Suggested Priority:** {row['priority']}"
            )

            with st.form(
                "staff_update_form"
            ):

                status = st.selectbox(
                    "Update Status",
                    STATUSES,
                    index=(
                        STATUSES.index(row["status"])
                        if row["status"] in STATUSES
                        else 0
                    )
                )

                staff = st.text_input(
                    "Staff / Officer Name",
                    value=row["assigned_to"] or ""
                )

                note = st.text_area(
                    "Staff Note",
                    value=row["staff_note"] or "",
                    placeholder=(
                        "Write what was checked or what action was taken."
                    )
                )

                resolution = st.text_area(
                    "Resolution",
                    value=row["resolution"] or ""
                )

                save = st.form_submit_button(
                    "💾 Save Update",
                    use_container_width=True
                )

            if save:

                update_complaint(
                    selected_id,
                    status,
                    staff,
                    note,
                    resolution,
                )

                st.success(
                    "Complaint updated successfully."
                )

                st.rerun()


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.title("📊 Hospital Complaint Analytics")

    df = get_all_complaints()

    if df.empty:

        st.info("No complaint data available yet.")

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Complaints",
            len(df)
        )

        c2.metric(
            "Resolved",
            len(
                df[
                    df.status.isin(
                        ["Resolved", "Closed"]
                    )
                ]
            )
        )

        c3.metric(
            "High / Immediate",
            len(
                df[
                    df.priority.isin(
                        ["High", "Immediate Attention"]
                    )
                ]
            )
        )

        a, b = st.columns(2)

        with a:

            st.subheader(
                "Complaints by Department"
            )

            st.bar_chart(
                df.department.value_counts()
            )

        with b:

            st.subheader(
                "Complaints by Category"
            )

            st.bar_chart(
                df.category.value_counts()
            )

        st.subheader(
            "Complaints by Status"
        )

        st.bar_chart(
            df.status.value_counts()
        )

        st.subheader(
            "Simple Quality Improvement Insight"
        )

        st.info(
            f"Most complaints are currently routed to "
            f"**{df.department.mode()[0]}**, and the most common "
            f"category is **{df.category.mode()[0]}**."
        )

        st.caption(
            "Analytics support management review. They are not proof "
            "of negligence or poor performance."
        )


# =========================================================
# ABOUT
# =========================================================

else:

    st.title("ℹ️ About This Project")

    st.markdown("""
## AI-Powered Hospital Complaint Agent

This version uses a **conversational AI workflow**.

Instead of giving the patient a long form, the AI can ask
follow-up questions one at a time.

### Example

Patient:

> Meri ammi emergency mein hain.

AI:

> Which hospital are you currently visiting?

Patient:

> Niazi Medical College.

AI:

> Approximately how long has she been waiting?

Patient:

> 2 hours.

AI:

> Thank you. I have enough information to prepare your complaint.

Then the system shows the patient a structured summary for confirmation.

### Main workflow

Patient
→ Conversation
→ AI understands
→ AI asks missing questions
→ AI creates structured complaint
→ Patient confirms
→ Ticket generated
→ Department routing
→ Human staff review
→ Action
→ Status updates
→ Escalation
→ Resolution
→ Analytics

### Important safety rule

The AI is an administrative complaint assistant.

It does NOT:
- diagnose
- prescribe
- decide treatment
- make legal judgments
- declare negligence
- replace doctors
- replace hospital management

For real deployment, the system would require authentication,
authorization, encryption, privacy controls, audit logs,
hospital-approved workflows, security testing and human oversight.
""")


st.markdown("""
<div class="app-footer">
    AI Hospital Complaint Agent • MVP Prototype<br>
    AI assists the process — authorized hospital staff make final decisions.
</div>
""", unsafe_allow_html=True)
