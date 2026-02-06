"""
Diagnostic Assistant - Streamlit Interface
Pure focus: Symptoms + Patient History → Diagnosis + Treatment
"""

import streamlit as st
import os
from pipeline import (
    load_or_create_vectorstore,
    DiagnosticEngine,
    PatientProfile,
    DOCS_PATH,
    OPENAI_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    GOOGLE_AVAILABLE,
    MODEL_RECOMMENDATIONS
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Medical Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .diagnosis-box {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .patient-info-card {
        background: #f0f7ff;
        border-left: 4px solid #2d5a87;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    .source-indicator {
        background: #e8f5e9;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .web-indicator {
        background: #fff3e0;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "engine" not in st.session_state:
    st.session_state.engine = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(hash(str(st.session_state)))[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = PatientProfile()
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# =============================================================================
# INITIALIZATION
# =============================================================================

@st.cache_resource
def initialize_engine():
    """Initialize the diagnostic engine."""
    try:
        if not os.path.exists(DOCS_PATH):
            os.makedirs(DOCS_PATH)
            return None, f"Created '{DOCS_PATH}' directory. Add medical PDFs and restart."
        
        if not os.listdir(DOCS_PATH):
            return None, f"'{DOCS_PATH}' is empty. Add medical PDF documents."
        
        vectorstore = load_or_create_vectorstore()
        engine = DiagnosticEngine(vectorstore)
        return engine, None
    
    except Exception as e:
        return None, str(e)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">🩺 Medical Diagnostic Assistant</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Symptoms → Diagnosis → Treatment Recommendations</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - PATIENT INFORMATION
# =============================================================================

with st.sidebar:
    st.header("🔧 Configuration")
    
    # Provider selection
    st.subheader("AI Provider")
    
    available_providers = []
    if OPENAI_AVAILABLE:
        available_providers.append("OpenAI (GPT-4o)")
    if ANTHROPIC_AVAILABLE:
        available_providers.append("Anthropic (Claude)")
    if GOOGLE_AVAILABLE:
        available_providers.append("Google (Gemini)")
    
    if not available_providers:
        st.error("No LLM providers installed. Run: pip install langchain-openai langchain-anthropic langchain-google-genai")
        st.stop()
    
    provider_choice = st.selectbox(
        "Select AI Provider",
        available_providers,
        help="Claude Sonnet 4: Best reasoning | GPT-4o: Highest benchmarks | Gemini: Fast & cheap"
    )
    
    # Map to provider code - UPDATED TO BEST 2025 MODELS
    provider_map = {
        "OpenAI (GPT-4o)": ("openai", "gpt-4o"),
        "Anthropic (Claude)": ("anthropic", "claude-sonnet-4-20250514"),
        "Google (Gemini)": ("google", "gemini-2.0-flash")
    }
    selected_provider, selected_model = provider_map[provider_choice]
    
    # API Key
    api_key = st.text_input(
        f"API Key",
        type="password",
        help=f"Enter your {selected_provider.upper()} API key"
    )
    
    # Model options - UPDATED 2025
    if selected_provider == "openai":
        model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    elif selected_provider == "anthropic":
        model_options = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    else:
        model_options = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    selected_model = st.selectbox("Model", model_options)
    
    st.markdown("---")
    
    st.header("👤 Patient Information")
    st.caption("Provide patient details for personalized diagnosis")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 
                              value=st.session_state.patient_profile.age or 0,
                              key="age_input")
    with col2:
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"],
                              index=0 if not st.session_state.patient_profile.gender else 
                              ["", "Male", "Female", "Other"].index(st.session_state.patient_profile.gender),
                              key="gender_input")
    
    weight = st.number_input("Weight (kg)", 0.0, 300.0, 
                             value=st.session_state.patient_profile.weight or 0.0,
                             key="weight_input")
    
    st.markdown("---")
    
    # Medical history
    st.subheader("📋 Medical History")
    
    conditions = st.text_area(
        "Existing Conditions",
        value=", ".join(st.session_state.patient_profile.conditions),
        placeholder="e.g., Diabetes, Hypertension, Asthma",
        help="Separate multiple conditions with commas",
        key="conditions_input"
    )
    
    allergies = st.text_area(
        "Allergies",
        value=", ".join(st.session_state.patient_profile.allergies),
        placeholder="e.g., Penicillin, Sulfa, Aspirin",
        help="Drug and other allergies",
        key="allergies_input"
    )
    
    medications = st.text_area(
        "Current Medications",
        value=", ".join(st.session_state.patient_profile.medications),
        placeholder="e.g., Metformin 500mg, Lisinopril 10mg",
        key="medications_input"
    )
    
    st.markdown("---")
    
    # Additional history
    st.subheader("📜 Additional History")
    
    family_history = st.text_area(
        "Family History",
        value=", ".join(st.session_state.patient_profile.family_history),
        placeholder="e.g., Heart disease (father), Diabetes (mother)",
        key="family_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        smoker = st.selectbox("Smoker?", ["Unknown", "No", "Yes"],
                              key="smoker_input")
    with col2:
        alcohol = st.selectbox("Alcohol Use", 
                               ["Unknown", "None", "Occasional", "Regular", "Heavy"],
                               key="alcohol_input")
    
    past_surgeries = st.text_area(
        "Past Surgeries",
        value=", ".join(st.session_state.patient_profile.past_surgeries),
        placeholder="e.g., Appendectomy 2015, Knee replacement 2020",
        key="surgeries_input"
    )
    
    # Update button
    if st.button("💾 Update Patient Profile", use_container_width=True):
        # Parse comma-separated fields
        def parse_list(text):
            if not text.strip():
                return []
            return [item.strip() for item in text.split(",") if item.strip()]
        
        st.session_state.patient_profile = PatientProfile(
            age=age if age > 0 else None,
            gender=gender if gender else None,
            weight=weight if weight > 0 else None,
            conditions=parse_list(conditions),
            allergies=parse_list(allergies),
            medications=parse_list(medications),
            family_history=parse_list(family_history),
            smoker=True if smoker == "Yes" else (False if smoker == "No" else None),
            alcohol_use=alcohol.lower() if alcohol != "Unknown" else None,
            past_surgeries=parse_list(past_surgeries)
        )
        
        # Update engine's patient session if initialized
        if st.session_state.engine:
            st.session_state.engine.patient_sessions[st.session_state.session_id] = st.session_state.patient_profile
        
        st.success("✓ Patient profile updated")
    
    # Show current profile summary
    st.markdown("---")
    st.subheader("📊 Current Profile")
    profile_str = st.session_state.patient_profile.to_string()
    if profile_str != "No patient history provided":
        st.markdown(f"""
        <div class="patient-info-card">
            {profile_str.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No patient information entered yet")

# =============================================================================
# MAIN AREA - INITIALIZE
# =============================================================================

# Check for API key
if not api_key:
    st.warning(f"👈 Enter your {selected_provider.upper()} API key in the sidebar to start")
    st.info("""
    **Which provider should you use?**
    
    | Provider | Best For | Cost |
    |----------|----------|------|
    | **OpenAI GPT-4o** | Highest USMLE benchmark accuracy (~92%) | ~$2.50-10/1M tokens |
    | **Claude Sonnet 4** | Best clinical reasoning & safety | ~$3-15/1M tokens |
    | **Gemini 2.0 Flash** | Fast, cheap, simple triage cases | ~$0.10/1M tokens |
    
    For medical diagnosis, **Claude Sonnet 4** or **GPT-4o** is recommended.
    """)
    st.stop()

# Initialize engine with selected provider
@st.cache_resource
def get_engine(provider: str, model: str, key: str):
    """Initialize diagnostic engine."""
    try:
        if not os.path.exists(DOCS_PATH):
            os.makedirs(DOCS_PATH)
            return None, f"Created '{DOCS_PATH}' directory. Add medical PDFs and restart."
        
        if not os.listdir(DOCS_PATH):
            return None, f"'{DOCS_PATH}' is empty. Add medical PDF documents."
        
        vectorstore = load_or_create_vectorstore()
        engine = DiagnosticEngine(
            provider=provider,
            model=model,
            api_key=key,
            vectorstore=vectorstore
        )
        return engine, None
    except Exception as e:
        return None, str(e)

engine, error = get_engine(selected_provider, selected_model, api_key)

if error:
    st.error(f"⚠️ {error}")
    st.stop()

st.session_state.engine = engine
st.session_state.engine.patient_sessions[st.session_state.session_id] = st.session_state.patient_profile

# Show current config
st.success(f"✓ Using **{selected_model}** ({selected_provider})")

# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source indicators for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            sources = message["sources"]
            cols = st.columns(4)
            with cols[0]:
                if sources.get("documents"):
                    st.markdown('<span class="source-indicator">📚 Documents Used</span>', unsafe_allow_html=True)
            with cols[1]:
                if sources.get("web"):
                    st.markdown('<span class="web-indicator">🌐 Web Search Used</span>', unsafe_allow_html=True)

# =============================================================================
# SYMPTOM INPUT
# =============================================================================

# Quick symptom templates
if not st.session_state.messages:
    st.markdown("### 💭 Describe the patient's symptoms")
    st.caption("Include: what symptoms, when they started, severity, and any patterns")
    
    st.markdown("**Example queries:**")
    examples = [
        "Patient has fever of 101°F, productive cough with yellow sputum, chest pain when breathing deeply, fatigue for 3 days",
        "Severe headache on one side, nausea, sensitivity to light, started this morning",
        "Burning sensation when urinating, frequent urination, lower abdominal discomfort for 2 days",
        "Joint pain and stiffness in both knees, worse in the morning, mild swelling"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"📝 {example[:50]}...", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

# Handle example query
if hasattr(st.session_state, 'example_query') and st.session_state.example_query:
    symptoms_input = st.session_state.example_query
    st.session_state.example_query = None
else:
    symptoms_input = st.chat_input("Describe the symptoms in detail...")

# =============================================================================
# PROCESS SYMPTOMS
# =============================================================================

if symptoms_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": symptoms_input
    })
    
    with st.chat_message("user"):
        st.markdown(symptoms_input)
    
    # Get diagnosis
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms and generating diagnosis..."):
            try:
                # Update engine with current patient profile
                st.session_state.engine.patient_sessions[st.session_state.session_id] = st.session_state.patient_profile
                
                # Get diagnosis
                result = st.session_state.engine.diagnose(
                    symptoms=symptoms_input,
                    session_id=st.session_state.session_id,
                    use_web_search=True
                )
                
                diagnosis = result["diagnosis"]
                
                # Show source indicators
                cols = st.columns(4)
                with cols[0]:
                    if result.get("document_context_used"):
                        st.markdown('<span class="source-indicator">📚 Documents Used</span>', unsafe_allow_html=True)
                with cols[1]:
                    if result.get("web_search_used"):
                        st.markdown('<span class="web-indicator">🌐 Web Search Used</span>', unsafe_allow_html=True)
                
                # Show patient context used
                if st.session_state.patient_profile.to_string() != "No patient history provided":
                    with st.expander("👤 Patient Profile Used for This Diagnosis"):
                        st.text(st.session_state.patient_profile.to_string())
                
                # Show diagnosis
                st.markdown(diagnosis)
                
                # Store in messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": diagnosis,
                    "sources": {
                        "documents": result.get("document_context_used"),
                        "web": result.get("web_search_used")
                    }
                })
                
            except Exception as e:
                error_msg = f"Error generating diagnosis: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": {}
                })

# =============================================================================
# CLEAR CHAT
# =============================================================================

if st.session_state.messages:
    if st.button("🗑️ Clear Conversation", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
