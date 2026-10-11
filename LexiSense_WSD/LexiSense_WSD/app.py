import os, re
import pandas as pd
import streamlit as st
import nltk

from src.wsd_engine import analyze, suggest_ambiguous_words
from src.wordnet_utils import synsets_for_word, synset_info, relation_graph_data
from src.visualizations import score_chart, dataset_word_chart
from utils.helpers import load_uploaded, results_to_csv

st.set_page_config(page_title="LexiSense | Word Sense Disambiguation", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def setup_nltk():
    resources = {"wordnet":"corpora/wordnet", "omw-1.4":"corpora/omw-1.4", "punkt":"tokenizers/punkt", "punkt_tab":"tokenizers/punkt_tab", "stopwords":"corpora/stopwords"}
    for pkg, path in resources.items():
        try: nltk.data.find(path)
        except LookupError:
            try: nltk.download(pkg, quiet=True)
            except Exception: pass
    return True
setup_nltk()

CSS = """
<style>
:root { --navy:#08152f; --blue:#2563eb; --purple:#7c3aed; --ink:#10203a; --muted:#64748b; --line:#e7edf6; }
.stApp { background: linear-gradient(135deg,#f7faff 0%,#eef3ff 48%,#f8f7ff 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#07142e 0%,#0d1f43 100%); border-right:1px solid #172e57; }
[data-testid="stSidebar"] * { color:#eaf2ff !important; }
.brand { padding:22px 18px 10px; }
.brand-title { font-size:28px; font-weight:850; letter-spacing:-1px; }
.brand-sub { color:#a8bce0 !important; font-size:12px; margin-top:3px; }
.hero { padding:48px 44px; border-radius:28px; background: radial-gradient(circle at 85% 15%,rgba(124,58,237,.28),transparent 28%), linear-gradient(120deg,#08152f,#102b5c); color:white; box-shadow:0 22px 55px rgba(20,39,80,.20); margin-bottom:28px; }
.hero h1 { font-size:52px; margin:0; letter-spacing:-2px; line-height:1; }
.hero p { font-size:18px; color:#cbd9f2; max-width:760px; margin:18px 0 0; }
.kicker { color:#8bb8ff; font-weight:800; letter-spacing:2px; font-size:12px; text-transform:uppercase; margin-bottom:12px; }
.card { background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:20px; padding:22px; box-shadow:0 10px 30px rgba(26,47,91,.07); height:100%; }
.card h3 { margin:8px 0 8px; color:#0b1d3a; }
.icon { font-size:28px; }
.section { margin:30px 0 14px; }
.section h2 { font-size:25px; letter-spacing:-.5px; margin-bottom:3px; }
.section p { color:var(--muted); }
.pipeline { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:16px; }
.step { background:white; border:1px solid var(--line); padding:12px 16px; border-radius:14px; font-weight:700; box-shadow:0 5px 15px rgba(25,45,90,.05); }
.arrow { color:#6d7fa0; font-size:18px; }
.result { background:linear-gradient(135deg,#fff,#f3f7ff); border:1px solid #d8e4fa; border-radius:24px; padding:26px; box-shadow:0 15px 40px rgba(35,62,110,.09); }
.result-label { color:#2563eb; font-weight:800; font-size:11px; letter-spacing:1.7px; }
.result-sense { font-size:34px; font-weight:850; color:#0b1d3a; margin:5px 0; }
.definition { font-size:17px; color:#40516d; }
.selected { border:2px solid #4f7df3 !important; background:linear-gradient(135deg,#f7faff,#eef4ff) !important; }
.sense-card { background:white; border:1px solid #e2e8f0; border-radius:18px; padding:18px; margin:10px 0; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; background:#e9f1ff; color:#2563eb; font-size:11px; font-weight:800; margin:2px; }
.footer { color:#9fb0cc; font-size:11px; text-align:center; padding:25px 0 5px; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:16px; padding:15px; box-shadow:0 6px 20px rgba(30,50,90,.05); }
.stButton > button { border-radius:12px; font-weight:750; border:1px solid #d8e2f2; }
.stDownloadButton > button { border-radius:12px; font-weight:750; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if 'dataset_results' not in st.session_state: st.session_state.dataset_results = pd.DataFrame()
if 'last_analysis' not in st.session_state: st.session_state.last_analysis = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-title">🧠 LexiSense</div><div class="brand-sub">Meaning through context</div></div>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["🏠 Home","🔍 Sense Analyzer","📂 Dataset Analyzer","📖 WordNet Explorer","📊 Analysis & Statistics","ℹ️ About Project"], label_visibility="collapsed")
    st.markdown('<div style="height:1px;background:#24406b;margin:18px 0"></div>', unsafe_allow_html=True)
    st.caption("WordNet-powered NLP research demo")
    st.markdown('<div class="footer">LexiSense • NLP Mini Project</div>', unsafe_allow_html=True)

def section(title, subtitle=""):
    st.markdown(f'<div class="section"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)

def safe_analyze(sentence, target, algorithm):
    try:
        return analyze(sentence, target, algorithm)
    except LookupError:
        return {"ok": False, "error": "WordNet data is not available yet. Please allow NLTK to download the WordNet corpus on first launch, then retry."}


def safe_synsets(word):
    try:
        return synsets_for_word(word)
    except LookupError:
        return []


def render_result(result):
    if not result or not result.get('ok'):
        st.error(result.get('error','Unable to analyze.') if result else 'No analysis yet.')
        return
    best=result['best']; info=best['info']
    c1,c2,c3=st.columns([1.5,1,1])
    with c1:
        st.markdown(f'''<div class="result"><div class="result-label">PREDICTED SENSE</div><div class="result-sense">{best['synset']}</div><div class="definition">{info['definition']}</div><div style="margin-top:13px">''' + ''.join(f'<span class="badge">{s}</span>' for s in info['synonyms'][:8]) + '</div></div>', unsafe_allow_html=True)
    with c2: st.metric("Match score", f"{best['score']:.2f}"); st.metric("Candidate senses", len(result['candidates']))
    with c3: st.metric("Comparative confidence", f"{result['confidence']*100:.1f}%"); st.metric("Context terms", len(result['context']))
    section("Why this sense?", "A transparent view of the lexical evidence used by Lesk.")
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><b>Important context</b><p>' + ' '.join(f'<span class="badge">{x}</span>' for x in result['context']) + '</p></div>', unsafe_allow_html=True)
    with b:
        matches=best['matches'] or ['No exact lexical overlap; score is tied/zero.']
        st.markdown('<div class="card"><b>Matching evidence</b><p>' + ' '.join(f'<span class="badge">{x}</span>' for x in matches) + '</p><p style="color:#64748b">The selected sense received the strongest contextual overlap under the chosen Lesk variant.</p></div>', unsafe_allow_html=True)
    section("Candidate senses", "Compare every WordNet candidate before the final selection.")
    for i,c in enumerate(result['candidates'],1):
        cls='selected' if c is best else ''
        badge='<span class="badge">✓ SELECTED</span>' if c is best else ''
        matches=', '.join(c['matches']) if c['matches'] else 'None'
        st.markdown(f'<div class="sense-card {cls}"><b>Sense {i}</b> <span style="float:right"><b>Score: {c["score"]:.2f}</b> {badge}</span><h3>{c["synset"]}</h3><p>{c["info"]["definition"]}</p><small><b>Context matches:</b> {matches}</small></div>', unsafe_allow_html=True)
    st.plotly_chart(score_chart(result['candidates']), use_container_width=True, config={'displayModeBar':False})
    with st.expander("📚 WordNet details for selected sense"):
        st.write("**Synonyms:**", ', '.join(info['synonyms']) or '—')
        st.write("**Hypernyms:**", ', '.join(info['hypernyms']) or '—')
        st.write("**Hyponyms:**", ', '.join(info['hyponyms']) or '—')
        st.write("**Examples:**")
        for ex in info['examples'] or ['No example sentence available.']: st.write('•', ex)

if page == "🏠 Home":
    st.markdown('<div class="hero"><div class="kicker">WordNet-powered NLP</div><h1>LexiSense</h1><p>Understanding Meaning Through Context</p><p style="font-size:15px">An interactive Word Sense Disambiguation system powered by WordNet and context-aware lexical analysis.</p></div>', unsafe_allow_html=True)
    cols=st.columns(3)
    cards=[('🧠','Context Understanding','Uses surrounding words to determine which meaning fits best.'),('📚','WordNet Intelligence','Explores synsets, definitions and lexical relationships.'),('📊','Explainable Results','Shows candidate senses, scores and matching evidence.')]
    for col,(icon,title,txt) in zip(cols,cards):
        with col: st.markdown(f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3><p>{txt}</p></div>',unsafe_allow_html=True)
    section("How It Works","From a sentence to an interpretable WordNet sense.")
    st.markdown('<div class="pipeline">' + '<span class="step">Sentence</span><span class="arrow">→</span><span class="step">Target Word</span><span class="arrow">→</span><span class="step">Candidate Senses</span><span class="arrow">→</span><span class="step">Context Matching</span><span class="arrow">→</span><span class="step">Predicted Sense</span></div>',unsafe_allow_html=True)
    section("Try the classic example","The same word can mean different things depending on context.")
    st.info('“The fisherman sat on the bank of the river.” → target: **bank** → the river-side sense should be favored by contextual evidence.')
    st.markdown('<div class="card"><b>Three ways to explore</b><br><br>🔍 Analyze one sentence &nbsp; • &nbsp; 📂 Process a CSV/TXT dataset &nbsp; • &nbsp; 📖 Explore WordNet relationships</div>',unsafe_allow_html=True)

elif page == "🔍 Sense Analyzer":
    section("Word Sense Analyzer","Discover which meaning of a word is intended by its context.")
    sentence=st.text_area("Enter your sentence", "The fisherman sat on the bank of the river.", height=130)
    c1,c2=st.columns([1,1])
    with c1:
        auto=suggest_ambiguous_words(sentence)
        target=st.text_input("Target word", value=auto[0] if auto else "bank")
        if auto: st.caption("Potentially ambiguous words: " + ', '.join(auto))
    with c2: algorithm=st.radio("Algorithm", ["Basic Lesk","Enhanced Lesk"], horizontal=True, index=1)
    if st.button("Analyze Meaning →", type="primary", use_container_width=True):
        if not sentence.strip() or not target.strip(): st.warning("Please enter both a sentence and target word.")
        elif not re.search(rf'\b{re.escape(target.strip())}\b', sentence, re.I): st.warning("The target word was not found in the sentence.")
        else:
            with st.spinner("Comparing WordNet senses…"): st.session_state.last_analysis=safe_analyze(sentence,target.strip(),algorithm)
    if st.session_state.last_analysis: render_result(st.session_state.last_analysis)

elif page == "📂 Dataset Analyzer":
    section("Dataset Analyzer","Run WSD over your own CSV or TXT file, then export the annotated results.")
    sample=pd.read_csv('data/sample_wsd_dataset.csv')
    source=st.radio("Dataset source",["Built-in sample dataset","Upload CSV / TXT"],horizontal=True)
    if source.startswith('Built-in'):
        df=sample.copy()
        st.caption(f"Ready-to-run demo dataset • {len(df)} sentences")
    else:
        up=st.file_uploader("Upload a CSV or TXT file",type=['csv','txt'])
        if not up: st.info("Upload a file to begin. CSV format: sentence,target_word. TXT uses one sentence per line."); st.stop()
        try: df=load_uploaded(up)
        except Exception as e: st.error(str(e)); st.stop()
    st.dataframe(df.head(12),use_container_width=True,hide_index=True)
    if st.button("Run WSD on Dataset →",type="primary",use_container_width=True):
        rows=[]
        bar=st.progress(0)
        for i,r in df.iterrows():
            sentence=str(r['sentence']); target=str(r['target_word']).strip()
            if not target:
                suggestions=suggest_ambiguous_words(sentence); target=suggestions[0] if suggestions else ''
            res=safe_analyze(sentence,target,"Enhanced Lesk") if target else {"ok":False}
            if res.get('ok'):
                b=res['best']; rows.append({'Sentence':sentence,'Target Word':target,'Predicted Sense':b['synset'],'Synonyms':', '.join(b['info']['synonyms']),'Definition':b['info']['definition'],'Score':b['score'],'Candidate Senses':len(res['candidates']),'Status':'Success'})
            else: rows.append({'Sentence':sentence,'Target Word':target,'Predicted Sense':'—','Synonyms':'','Definition':res.get('error','No sense found'),'Score':0,'Candidate Senses':0,'Status':'Failed'})
            bar.progress((i+1)/len(df))
        st.session_state.dataset_results=pd.DataFrame(rows)
    if not st.session_state.dataset_results.empty:
        out=st.session_state.dataset_results
        section("Dataset Summary","A compact overview of the current processing run.")
        m=st.columns(5); vals=[len(out),out['Target Word'].nunique(),out['Candidate Senses'].mean(),out['Score'].mean(),(out['Status']=='Success').sum()]
        for col,label,val in zip(m,['Sentences','Unique words','Avg candidates','Avg score','Successful'],vals): col.metric(label, f'{val:.1f}' if isinstance(val,float) else val)
        st.dataframe(out,use_container_width=True,hide_index=True)
        st.download_button("⬇ Download Results CSV",results_to_csv(out.to_dict('records')),"lexisense_wsd_results.csv","text/csv")

elif page == "📖 WordNet Explorer":
    section("WordNet Explorer","Inspect senses and lexical relationships without dumping raw corpus output.")
    word=st.text_input("Enter a word", "bank").strip().lower()
    if word:
        syns=safe_synsets(word)
        if not syns: st.warning("WordNet could not find a sense for this word. Try another target word.")
        else:
            st.markdown(f'<div class="card"><b>{word}</b> has <span class="badge">{len(syns)} synsets</span> in WordNet.</div>',unsafe_allow_html=True)
            for i,s in enumerate(syns):
                info=synset_info(s)
                with st.expander(f"{i+1:02d}  {s.name()} — {s.definition()}", expanded=(i==0)):
                    a,b,c=st.columns(3)
                    a.write("**Synonyms**"); a.write(', '.join(info['synonyms']) or '—')
                    b.write("**Hypernyms**"); b.write(', '.join(info['hypernyms']) or '—')
                    c.write("**Hyponyms**"); c.write(', '.join(info['hyponyms']) or '—')
                    if info['antonyms']: st.write("**Antonyms:**", ', '.join(info['antonyms']))
                    st.write("**Examples:**", ' • '.join(info['examples']) or 'No examples available.')
            section("Relationship map","A compact visual view of nearby WordNet concepts.")
            nodes,edges=relation_graph_data(word)
            # Plotly scatter network approximation, kept dependency-light.
            import plotly.graph_objects as go
            n=len(nodes); xs=[0]; ys=[0]
            for i in range(1,n):
                import math
                angle=2*math.pi*(i-1)/max(1,n-1); xs.append(math.cos(angle)*(1 if i<7 else 1.6)); ys.append(math.sin(angle)*(1 if i<7 else 1.6))
            fig=go.Figure()
            for a,b,rel in edges:
                ia=next(i for i,x in enumerate(nodes) if x['id']==a); ib=next(i for i,x in enumerate(nodes) if x['id']==b)
                fig.add_trace(go.Scatter(x=[xs[ia],xs[ib]],y=[ys[ia],ys[ib]],mode='lines',line=dict(width=1),hoverinfo='skip',showlegend=False))
            fig.add_trace(go.Scatter(x=xs,y=ys,mode='markers+text',text=[x['label'] for x in nodes],textposition='top center',marker=dict(size=16),hovertext=[x['group'] for x in nodes],hoverinfo='text',showlegend=False))
            fig.update_layout(height=500,template='plotly_white',xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

elif page == "📊 Analysis & Statistics":
    section("Analysis & Statistics","Explore patterns from the built-in dataset or your latest processing run.")
    df=st.session_state.dataset_results if not st.session_state.dataset_results.empty else pd.read_csv('data/sample_wsd_dataset.csv').assign(**{'Target Word':lambda x:x['target_word']})
    if 'Predicted Sense' not in df.columns:
        # Run a compact analysis for the sample automatically.
        rows=[]
        for _,r in df.iterrows():
            res=safe_analyze(r['sentence'],r['target_word'],"Enhanced Lesk")
            if res.get('ok'): rows.append({'Sentence':r['sentence'],'Target Word':r['target_word'],'Predicted Sense':res['best']['synset'],'Score':res['score'],'Candidate Senses':len(res['candidates'])})
        df=pd.DataFrame(rows)
    if not df.empty:
        m=st.columns(4)
        m[0].metric('Total Sentences',len(df)); m[1].metric('Unique Words',df['Target Word'].nunique()); m[2].metric('Average Candidate Senses',f"{df['Candidate Senses'].mean():.1f}"); m[3].metric('Average WSD Score',f"{df['Score'].mean():.2f}")
        st.plotly_chart(dataset_word_chart(df),use_container_width=True,config={'displayModeBar':False})
        section("Observations","Automatically generated from the processed records.")
        for word,g in df.groupby('Target Word'):
            senses=g['Predicted Sense'].nunique()
            if len(g)>1 and senses>1: st.info(f"**{word}** appears in {len(g)} contexts and receives {senses} different predicted senses, demonstrating contextual ambiguity.")
        st.dataframe(df,use_container_width=True,hide_index=True)

elif page == "ℹ️ About Project":
    section("About LexiSense","A college-level NLP demonstration of Word Sense Disambiguation with WordNet.")
    st.markdown('<div class="card"><h3>Methodology</h3><p>LexiSense retrieves candidate WordNet synsets for the target word, normalizes the surrounding context, compares it against synset lexical signatures, and selects the highest-overlap sense.</p><h3>Basic Lesk</h3><p>Scores overlap between context terms and each synset signature built from its definition, lemmas and examples.</p><h3>Enhanced Lesk</h3><p>Extends the signature with hypernym/hyponym definitions and gives a small additional weight to lemma matches.</p><h3>Important interpretation</h3><p>The displayed “confidence” is a comparative score across candidate senses, not a calibrated probability.</p></div>',unsafe_allow_html=True)
    st.write('**Local-first:** no API keys or external inference services are required.')
