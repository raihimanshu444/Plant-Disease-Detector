import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from reportlab.pdfgen import canvas
import io
import base64

# --- Page Setup ---
st.set_page_config(page_title="FarmAssist AI", page_icon="🌾", layout="wide")

# --- Load CSS ---
st.markdown("""
<style>
body, .stApp {
    background-color: #f0fff0;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #2c6e49;
}
.stButton>button, .stDownloadButton>button {
    background-color: #2c6e49;
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 10px 20px;
}
.stDownloadButton>button:hover {
    background-color: #25603d;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}
.supplement-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# --- Model Class ---
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 39)
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out


# --- Load Model & Data ---
@st.cache_data
def load_data():
    disease_df = pd.read_csv("disease_info.csv", encoding="ISO-8859-1")
    supplement_df = pd.read_csv("supplement_info.csv")

    # Create normalized disease names for matching
    supplement_df['normalized_disease'] = supplement_df['disease_name'].str.lower().str.replace('___', '_').str.replace(
        '__', '_').str.strip()

    return disease_df, supplement_df


disease_df, supplement_df = load_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --- Improved Disease-Supplement Matching ---
import difflib

def get_supplement_info(disease_name):
    # 1) Normalize incoming name
    normalized = (
        disease_name
        .lower()
        .replace('___', '_')
        .replace('__', '_')
        .strip()
    )
    # 2) Exact
    supp = supplement_df[supplement_df['normalized_disease'] == normalized]
    if not supp.empty:
        return supp

    # 3) Contains full normalized
    supp = supplement_df[supplement_df['normalized_disease'].str.contains(normalized, na=False)]
    if not supp.empty:
        return supp

    # 4) Contains last two tokens
    parts = normalized.split('_')
    if len(parts) >= 2:
        key = '_'.join(parts[-2:])
        supp = supplement_df[supplement_df['normalized_disease'].str.contains(key, na=False)]
        if not supp.empty:
            return supp

    # 5) Fuzzy match
    all_norms = supplement_df['normalized_disease'].tolist()
    close = difflib.get_close_matches(normalized, all_norms, n=1, cutoff=0.6)
    if close:
        return supplement_df[supplement_df['normalized_disease'] == close[0]]

    return pd.DataFrame()  # still nothing


# --- PDF Generator ---
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(display_name, desc, steps, supplement, confidence, lang):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    subtitle = styles["Heading3"]
    normal = styles["BodyText"]

    flowables = []
    # Title
    flowables.append(Paragraph("🌾 FarmAssist AI Report", title_style))
    flowables.append(Spacer(1, 12))

    # Disease & Confidence
    flowables.append(Paragraph(f"<b>Disease:</b> {display_name}", normal))
    flowables.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", normal))
    flowables.append(Spacer(1, 12))

    # Description
    flowables.append(Paragraph("Description:", subtitle))
    for line in desc.splitlines():
        flowables.append(Paragraph(line, normal))
    flowables.append(Spacer(1, 12))

    # Treatment Steps
    flowables.append(Paragraph("Treatment Steps:", subtitle))
    step_items = [ListItem(Paragraph(s.strip(), normal))
                  for s in steps.splitlines() if s.strip()]
    flowables.append(ListFlowable(step_items, bulletType="bullet"))
    flowables.append(Spacer(1, 12))

    # Supplements (if any)
    if not supplement.empty:
        flowables.append(Paragraph("Suggested Supplement:", subtitle))
        supp_items = [
            ListItem(Paragraph(f"{row['supplement name']} (Buy: {row['buy link']})", normal))
            for _, row in supplement.iterrows()
        ]
        flowables.append(ListFlowable(supp_items, bulletType="bullet"))
        flowables.append(Spacer(1, 12))

    # Footer
    flowables.append(Paragraph("Generated by FarmAssist AI", styles["Italic"]))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

# --- Language Selector ---
lang = st.selectbox("🌍 Choose your language / अपनी भाषा चुनें", ["English", "Hindi"])

# Save current language choice in session state
st.session_state["lang"] = lang

# Language dictionaries
L = {
    "English": {
        "home_title": "Welcome to FarmAssist AI",
        'home_description': 'This AI-powered app helps farmers quickly identify plant diseases and get treatment recommendations.',
        'image_caption': 'AI in Agriculture',
        'how_it_works': 'How It Works',
        'step1_title': 'Upload Photo',
        'step1_text': 'Capture or upload clear photos of plant leaves',
        'step2_title': 'AI Analysis',
        'step2_text': 'Our neural network analyzes plant health in seconds',
        'step3_title': 'Get Solutions',
        'step3_text': 'Immediate diagnosis and organic treatment options',
        "upload": "📷 Upload a plant leaf image",
        "detect": "Detect Disease",
        "supplements": "Supplements & Products",
        "result": "Diagnosis",
        "confidence": "Confidence",
        "description": "Description",
        "treatment": "Treatment Steps",
        "download_text": "📝 Download TXT Report",
        "download_pdf": "📄 Download PDF Report",
        "search_placeholder": "Search by disease or product",
        "no_results": "No results found.",
        "footer": "Developed by ❤️ FarmAssist AI | Empowering Farmers with AI",
        "suggested_supplement": "💊 Suggested Supplement",
        "no_supplement": "No supplement found for this disease."
    },
    "Hindi": {
        "home_title": "FarmAssist AI में आपका स्वागत है",
        'home_description': 'यह एआई-संचालित ऐप किसानों को पौधों के रोगों की त्वरित पहचान और उपचार सिफारिशें प्रदान करता है।',
        'image_caption': 'कृषि में कृत्रिम बुद्धिमत्ता',
        'how_it_works': 'यह कैसे काम करता है',
        'step1_title': 'फोटो अपलोड करें',
        'step1_text': 'पौधों के पत्तों की स्पष्ट तस्वीरें खींचें या अपलोड करें',
        'step2_title': 'एआई विश्लेषण',
        'step2_text': 'हमारा न्यूरल नेटवर्क सेकंडों में पौधों का विश्लेषण करता है',
        'step3_title': 'समाधान पाएं',
        'step3_text': 'तत्काल निदान और जैविक उपचार विकल्प',
        "upload": "📷 पत्ते की फोटो अपलोड करें",
        "detect": "बीमारी पहचानें",
        "supplements": "उर्वरक और उत्पाद",
        "result": "रोग पहचान",
        "confidence": "विश्वास स्तर",
        "description": "विवरण",
        "treatment": "इलाज के उपाय",
        "download_text": "📝 रिपोर्ट (TXT) डाउनलोड करें",
        "download_pdf": "📄 रिपोर्ट (PDF) डाउनलोड करें",
        "search_placeholder": "रोग या उत्पाद खोजें",
        "no_results": "कोई परिणाम नहीं मिला।",
        "footer": "❤️ FarmAssist AI द्वारा विकसित | किसानों को एआई से सशक्त बना रहे हैं",
        "suggested_supplement": "💊 सुझावित उर्वरक",
        "no_supplement": "इस रोग के लिए कोई उर्वरक जानकारी नहीं मिली।"
    }
}

# Tabs
tab1, tab2, tab3 = st.tabs([f"🏠 {L[lang]['home_title']}", f"🧠 {L[lang]['detect']}", f"💊 {L[lang]['supplements']}"])

# --- Home Tab ---
with tab1:
    st.markdown("""
    <style>
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background: white;
            margin-bottom: 20px;
        }
        .step-card {
            padding: 15px;
            text-align: center;
            background: #f8f9fa;
            border-radius: 10px;
            height: 200px;
        }
        .step-card h4 {
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(f"🌿 {L[lang]['home_title']}")
    st.markdown(L[lang]['home_description'])

    # Using a different plant image
    # st.image("https://images.unsplash.com/photo-1604876000906-d10f2c1a6c00",
             # use_container_width=True,
             # caption=L[lang]['image_caption'])

    st.subheader(L[lang]['how_it_works'])

    # Improved How it Works section
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="step-card">
            <h4>📸<br>{}</h4>
            <p>{}</p>
        </div>
        """.format(
            L[lang]['step1_title'],
            L[lang]['step1_text']
        ), unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
        <div class="step-card">
            <h4>🤖<br>{}</h4>
            <p>{}</p>
        </div>
        """.format(
            L[lang]['step2_title'],
            L[lang]['step2_text']
        ), unsafe_allow_html=True)

    with cols[2]:
        st.markdown("""
        <div class="step-card">
            <h4>💊<br>{}</h4>
            <p>{}</p>
        </div>
        """.format(
            L[lang]['step3_title'],
            L[lang]['step3_text']
        ), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Disease Detection Tab ---
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(f"🧠 {L[lang]['detect']}")

    uploaded_file = st.file_uploader(L[lang]["upload"], type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            tensor = transform(image).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            idx = pred.item()
            confidence = conf.item() * 100

            name = disease_df.loc[idx, "disease_name"].strip()
            display_name = name.replace("___", " ").replace("_", " : ")
            desc = disease_df.loc[idx, "description"]
            steps = disease_df.loc[idx, "Possible Steps"]

            # Get supplement info using improved matching
            supplement = get_supplement_info(name)

        st.success(f"✅ {L[lang]['result']}: {display_name}")
        st.markdown(f"**{L[lang]['confidence']}:** `{confidence:.2f}%`")
        st.markdown(f"### 📌 {L[lang]['description']}")
        st.write(desc)
        st.markdown(f"### 🛠 {L[lang]['treatment']}")
        st.write(steps)

        st.markdown(f"### {L[lang]['suggested_supplement']}")
        if not supplement.empty:
            for _, row in supplement.iterrows():
                with st.container():
                    st.markdown('<div class="supplement-card">', unsafe_allow_html=True)
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if pd.notnull(row['supplement image']):
                            st.image(row['supplement image'], width=100)
                    with cols[1]:
                        st.markdown(f"**{row['supplement name']}**")
                        st.markdown(f"[🛒 {'Buy Now' if lang == 'English' else 'अभी खरीदें'}]({row['buy link']})")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(L[lang]['no_supplement'])

        col1, col2 = st.columns(2)
        with col1:
            # Build a nicely formatted multiline text report
            bullet_steps = "\n".join(f"- {line}" for line in steps.splitlines() if line.strip())
            if not supplement.empty:
                supplement_lines = "\n".join(
                    f"- {row['supplement name']} (Buy: {row['buy link']})"
                    for _, row in supplement.iterrows()
                )
            else:
                supplement_lines = "-"

            txt_report = f"""
        FarmAssist AI Report
        =====================

        Disease: {display_name}
        Confidence: {confidence:.2f}%

        Description:
        {desc}

        Treatment Steps:
        {bullet_steps}

        {"Suggested Supplement:" if not supplement.empty else "No Supplement Found"}
        {supplement_lines}
        """.strip()

            st.download_button(
                L[lang]["download_text"],
                txt_report,
                file_name="farmassist_report.txt",
            )

        with col2:
            pdf_buf = generate_pdf(display_name, desc, steps, supplement, confidence, lang)
            b64 = base64.b64encode(pdf_buf.read()).decode()
            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">'
                f'<button style="width:100%;background:#2c6e49;color:white;padding:10px;border:none;border-radius:8px;">{L[lang]["download_pdf"]}</button>'
                '</a>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Supplement Tab ---
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(f"💊 {L[lang]['supplements']}")
    query = st.text_input("🔍 " + L[lang]['search_placeholder'])
    filtered = supplement_df
    if query:
        filtered = supplement_df[
            supplement_df['disease_name'].str.contains(query, case=False, na=False) |
            supplement_df['supplement name'].str.contains(query, case=False, na=False)
            ]
    if len(filtered) > 0:
        for _, row in filtered.iterrows():
            with st.container():
                st.markdown('<div class="supplement-card">', unsafe_allow_html=True)
                cols = st.columns([1, 3])
                with cols[0]:
                    if pd.notnull(row['supplement image']):
                        st.image(row['supplement image'], width=100)
                with cols[1]:
                    st.markdown(f"**{row['disease_name'].replace('___', ' ').replace('_', ' ')}**")
                    st.markdown(f"**Product:** {row['supplement name']}")
                    st.markdown(f"[🛒 {'Buy Now' if lang == 'English' else 'अभी खरीदें'}]({row['buy link']})")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(L[lang]['no_results'])

    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown(f"""
<hr>
<div style='text-align:center;color:#888;font-size:14px;'>
    {L[lang]['footer']}
</div>
""", unsafe_allow_html=True)
