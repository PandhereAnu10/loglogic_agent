import gradio as gr
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from groq import Groq
import os

# ============================================================
# SETUP
# ============================================================
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

#if not GROQ_API_KEY:
#    raise ValueError("GROQ_API_KEY environment variable not set")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY not set in Hugging Face Secrets")
    return Groq(api_key=api_key)

client = get_groq_client()

#client = Groq(api_key=GROQ_API_KEY)
print("Groq client initialized")


# ============================================================
# ANOMALY DETECTION
# ============================================================
def detect_anomalies(df):
    """Use Isolation Forest to find suspicious logs"""
    if len(df) < 10:
        return df
    
    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(df['message'])
    detector = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = detector.fit_predict(X)
    suspicious_logs = df[df['anomaly'] == -1].copy()
    
    print(f"Found {len(suspicious_logs)} suspicious logs out of {len(df)}")
    return suspicious_logs


# ============================================================
# LLM ANALYSIS
# ============================================================
def analyze_logs_with_llm(suspicious_logs):
    """Agentic analysis with chain-of-thought"""
    log_summary = suspicious_logs[['timestamp', 'level', 'service', 'message']].to_string()
    
    prompt = f"""You are a Senior SRE investigating a system incident. Analyze these production logs using multi-step reasoning:
{log_summary}
Think step-by-step:
STEP 1 - PATTERN RECOGNITION:
What patterns do you notice?
STEP 2 - GENERATE HYPOTHESES:
List 3 possible root causes.
STEP 3 - EVALUATE EVIDENCE:
What evidence supports each hypothesis?
STEP 4 - ROOT CAUSE CONCLUSION:
Which hypothesis is most likely and why?
STEP 5 - IMPACT ASSESSMENT:
- Affected Services
- Urgency: Critical/High/Medium/Low
STEP 6 - RECOMMENDED FIX:
Exact commands or code changes needed.
STEP 7 - VERIFICATION:
What would you check to verify this diagnosis?
Be concise but show your reasoning."""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=800
    )
    
    return response.choices[0].message.content


# ============================================================
# FIX GENERATION
# ============================================================
def generate_fix(diagnosis, suspicious_logs):
    """Generate fix with self-verification"""
    prompt_fix = f"""Based on this diagnosis:
{diagnosis}
Generate exact commands to fix this issue.
Provide:
1. Immediate fix commands
2. Long-term prevention
3. Rollback plan"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_fix}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=400
    )
    
    proposed_fix = response.choices[0].message.content
    
    # Self-verification
    log_summary = suspicious_logs[['timestamp', 'level', 'service', 'message']].to_string()
    prompt_verify = f"""You proposed: {proposed_fix}
Self-verify:
1. Does this fix the root cause?
2. What are the risks?
3. Confidence (0-100%)?"""

    verification = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_verify}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=300
    )
    
    return f"{proposed_fix}\n\n---\n**VERIFICATION:**\n{verification.choices[0].message.content}"


# ============================================================
# MAIN AGENT
# ============================================================
def loglogic_agent(log_dataframe):
    """Complete agentic diagnostic agent"""
    print("LogLogic Agent Starting...")
    
    suspicious = detect_anomalies(log_dataframe)
    
    if len(suspicious) == 0:
        return {
            "diagnosis": "**No critical issues detected!**\n\nYour system appears healthy.",
            "fix": "No action needed. System is operating normally."
        }
    
    print("Analyzing with multi-step reasoning...")
    diagnosis = analyze_logs_with_llm(suspicious)
    
    print("Self-verifying diagnosis...")
    verification_prompt = f"""Review your diagnosis: {diagnosis}
1. Alternative explanations?
2. Confidence (0-100%)?
3. What's missing?"""

    verification = client.chat.completions.create(
        messages=[{"role": "user", "content": verification_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=300
    )
    
    self_check = verification.choices[0].message.content
    
    print("🔧 Generating fix...")
    fix = generate_fix(diagnosis, suspicious)
    
    reasoning = f"## DIAGNOSIS\n\n{diagnosis}\n\n---\n\n## 🔍 SELF-VERIFICATION\n\n{self_check}"
    
    print("Analysis complete!")
    return {"diagnosis": reasoning, "fix": fix}


# ============================================================
# FILE PROCESSING
# ============================================================
def process_uploaded_file(file):
    """Handle uploaded log files"""
    if file is None:
        return "Please upload a log file", "No file selected", "Upload a file to begin"
    
    try:
        print(f"Processing: {file.name}")
        
        logs = []
        try:
            df_logs = pd.read_csv(file.name)
            print(f"Loaded CSV: {len(df_logs)} rows")
            
            if 'message' not in df_logs.columns:
                raise ValueError("CSV needs 'message' column")
        except:
            print("Parsing as text file...")
            with open(file.name, 'r', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    match = re.match(r'\[(.*?)\] \[(.*?)\](.*)', line)
                    if match:
                        timestamp, level, message = match.groups()
                        logs.append({
                            'timestamp': timestamp.strip(),
                            'level': level.strip().upper(),
                            'service': 'system',
                            'message': message.strip()
                        })
                    elif len(line.split()) >= 2:
                        logs.append({
                            'timestamp': 'unknown',
                            'level': 'ERROR' if 'error' in line.lower() else 'INFO',
                            'service': 'system',
                            'message': line.strip()
                        })
            
            df_logs = pd.DataFrame(logs)
            print(f"Parsed text file: {len(df_logs)} logs")
        
        if len(df_logs) == 0:
            return "❌ No valid logs found", "File appears empty", "0 logs parsed"
        
        for col in ['timestamp', 'level', 'service', 'message']:
            if col not in df_logs.columns:
                df_logs[col] = 'unknown'
        
        preview = f"### Successfully Loaded {len(df_logs)} Logs\n\n"
        preview += f"**Log Levels:** {dict(df_logs['level'].value_counts())}\n\n"
        preview += "**Sample (First 10 Logs):**\n```\n"
        preview += df_logs.head(10)[['timestamp', 'level', 'message']].to_string()
        preview += "\n```"
        
        result = loglogic_agent(df_logs)
        
        return result['diagnosis'], result['fix'], preview
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return f"## ERROR\n\n```\n{str(e)}\n```", "Processing failed", "Check diagnosis for details"


# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(), title="LogLogic Agent") as demo:
    gr.Markdown(
        """
        # LogLogic: Autonomous Diagnostic Agent
        
        Upload your server logs and get instant AI-powered diagnosis with multi-step agentic reasoning.
        
        **Supported formats:** `.log`, `.txt`, `.csv` (Apache, Nginx, generic text logs)
        """
    )
    
    with gr.Row():
        file_input = gr.File(
            label="📁 Upload Your Log File",
            file_types=[".log", ".txt", ".csv"]
        )
    
    analyze_btn = gr.Button("Analyze Logs", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            diagnosis_output = gr.Markdown(label="Diagnosis")
        with gr.Column():
            fix_output = gr.Markdown(label="Recommended Fix")
    
    with gr.Row():
        preview_output = gr.Markdown(label="Log Preview")
    
    gr.Markdown(
        """
        ---
        
        ### How it works:
        1. **ML Filtering:** Isolation Forest detects anomalies
        2. **Agentic Analysis:** Multi-step chain-of-thought reasoning
        3. **Self-Verification:** Agent validates its own diagnosis
        4. **Fix Generation:** Executable commands with safety checks
        
        Built with [Groq](https://groq.com/) • [GitHub](https://github.com/)
        """
    )
    
    analyze_btn.click(
        fn=process_uploaded_file,
        inputs=file_input,
        outputs=[diagnosis_output, fix_output, preview_output]
    )

# LAUNCH

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LogLogic: Autonomous Diagnostic Agent")
    print("="*60 + "\n")
    
    demo.queue()  # Enable queue for HF Spaces
    demo.launch(
        server_name="0.0.0.0",
        # server_port=7860,
        show_error=True
    )
