"""
app.py  –  RecoSense · Streamlit UI
AIE425 – Alamein International University
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data.loader              import load_reviews, dataset_stats
from recommenders.collaborative  import CollaborativeFilteringEngine, METHODS as CF_METHODS
from recommenders.content_based  import ContentBasedEngine,           METHODS as CB_METHODS
from recommenders.knowledge_based import KnowledgeBasedEngine,        RANKING_OPTIONS
from evaluation.evaluator     import Evaluator

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RecoSense · AIE425",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean dark theme + rich animations
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ────────────────────────────────────────────────────── */
:root {
  --bg:       #0d0d14;
  --surface:  #13131e;
  --card:     #1a1a28;
  --card2:    #1f1f30;
  --border:   #2e2e45;
  --accent:   #6c63ff;
  --accentL:  #8b84ff;
  --green:    #3dd68c;
  --pink:     #ff6b9d;
  --amber:    #ffb347;
  --text:     #f0f0f8;
  --text2:    #b0b0c8;
  --muted:    #6a6a85;
  --radius:   14px;
  --font:     'Plus Jakarta Sans', sans-serif;
  --mono:     'JetBrains Mono', monospace;
}

/* ── Reset & base ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main { background: var(--bg) !important; }

[data-testid="stApp"] * {
  font-family: var(--font) !important;
  color: var(--text);
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide default Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

/* ═══════════════════════════════════════════════════════════════════
   ANIMATIONS  (fixed & extended)
   ═══════════════════════════════════════════════════════════════════ */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(22px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes fadeDown {
  from { opacity: 0; transform: translateY(-16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideRight {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideLeft {
  from { opacity: 0; transform: translateX(20px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.88); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes glow {
  0%,100% { box-shadow: 0 0 0 0 rgba(108,99,255,.3); }
  50%      { box-shadow: 0 0 28px 6px rgba(108,99,255,.5); }
}
@keyframes glowPink {
  0%,100% { box-shadow: 0 0 0 0 rgba(255,107,157,.2); }
  50%      { box-shadow: 0 0 20px 4px rgba(255,107,157,.4); }
}
@keyframes shimmer {
  0%   { background-position: -600px 0; }
  100% { background-position:  600px 0; }
}
@keyframes barGrow {
  from { width: 0; opacity: 0; }
  to   { width: var(--w); opacity: 1; }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50%      { opacity: .6; transform: scale(0.96); }
}
@keyframes floatY {
  0%,100% { transform: translateY(0); }
  50%      { transform: translateY(-6px); }
}
@keyframes borderRotate {
  from { background-position: 0% 50%; }
  to   { background-position: 100% 50%; }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(10px) scale(0.9); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes ripple {
  0%   { transform: scale(0); opacity: .6; }
  100% { transform: scale(4); opacity: 0; }
}
@keyframes typewriter {
  from { width: 0; }
  to   { width: 100%; }
}
@keyframes blink {
  0%,100% { border-color: var(--accent); }
  50%      { border-color: transparent; }
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes orbFloat {
  0%   { transform: translate(0, 0) scale(1); }
  33%  { transform: translate(30px, -20px) scale(1.05); }
  66%  { transform: translate(-20px, 15px) scale(0.97); }
  100% { transform: translate(0, 0) scale(1); }
}
@keyframes cardEntrance {
  from { opacity: 0; transform: translateY(30px) scale(0.95); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes lineExpand {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}
@keyframes dotBounce {
  0%,80%,100% { transform: scale(0); }
  40%         { transform: scale(1); }
}

/* ═══════════════════════════════════════════════════════════════════
   ANIMATED BACKGROUND ORBS (hero)
   ═══════════════════════════════════════════════════════════════════ */
.hero-orbs {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  border-radius: inherit;
}
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(60px);
  opacity: .18;
  animation: orbFloat linear infinite;
}
.orb-1 {
  width: 300px; height: 300px;
  background: var(--accent);
  top: -80px; left: 10%;
  animation-duration: 12s;
}
.orb-2 {
  width: 250px; height: 250px;
  background: var(--pink);
  top: 20px; right: 5%;
  animation-duration: 16s;
  animation-delay: -4s;
}
.orb-3 {
  width: 200px; height: 200px;
  background: var(--green);
  bottom: -40px; left: 40%;
  animation-duration: 20s;
  animation-delay: -8s;
}

/* ═══════════════════════════════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════════════════════════════ */
.hero-wrap {
  text-align: center;
  padding: 70px 20px 50px;
  position: relative;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(108,99,255,.06) 0%, transparent 100%);
  border-radius: 24px;
  border: 1px solid var(--border);
  margin-bottom: 32px;
}
.hero-logo {
  font-size: clamp(2.8rem, 7vw, 5rem);
  font-weight: 800;
  letter-spacing: -3px;
  line-height: 1;
  background: linear-gradient(130deg, #ffffff 0%, var(--accentL) 40%, var(--pink) 80%, var(--amber) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeUp .8s cubic-bezier(.16,1,.3,1) both, gradientShift 4s ease infinite;
}
.hero-tagline {
  font-size: .85rem;
  font-weight: 600;
  letter-spacing: .25em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin: 12px 0 20px;
  animation: fadeUp .8s cubic-bezier(.16,1,.3,1) .12s both;
}
.hero-pill {
  display: inline-block;
  background: linear-gradient(135deg, var(--accent), var(--pink));
  background-size: 200% auto;
  color: #fff !important;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .15em;
  text-transform: uppercase;
  padding: 5px 20px;
  border-radius: 999px;
  animation: fadeUp .8s cubic-bezier(.16,1,.3,1) .24s both, glow 3s ease 1.2s infinite, gradientShift 3s ease infinite;
}
.hero-instruction {
  color: var(--text2) !important;
  font-size: .95rem;
  line-height: 1.9;
  margin-top: 36px;
  animation: fadeUp .8s cubic-bezier(.16,1,.3,1) .36s both;
}
.hero-instruction code {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 2px 10px;
  font-family: var(--mono) !important;
  font-size: .82rem;
  color: var(--green) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   STAT CARDS (hero)
   ═══════════════════════════════════════════════════════════════════ */
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 52px;
  animation: fadeUp .8s cubic-bezier(.16,1,.3,1) .48s both;
}
.hstat {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px 30px;
  min-width: 150px;
  text-align: center;
  transition: transform .3s cubic-bezier(.34,1.56,.64,1), border-color .25s, box-shadow .25s;
  position: relative;
  overflow: hidden;
  cursor: default;
}
.hstat::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--pink), var(--green));
  background-size: 200% auto;
  opacity: 0;
  transition: opacity .25s;
  animation: gradientShift 3s ease infinite;
}
.hstat::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% 0%, rgba(108,99,255,.08), transparent 70%);
  opacity: 0;
  transition: opacity .25s;
}
.hstat:hover { transform: translateY(-6px) scale(1.02); border-color: var(--accent); box-shadow: 0 12px 40px rgba(108,99,255,.2); }
.hstat:hover::before { opacity: 1; }
.hstat:hover::after  { opacity: 1; }
.hstat-num {
  font-size: 2.4rem;
  font-weight: 800;
  color: var(--accentL) !important;
  line-height: 1;
  animation: countUp .6s cubic-bezier(.16,1,.3,1) both;
}
.hstat:nth-child(1) .hstat-num { animation-delay: .5s; }
.hstat:nth-child(2) .hstat-num { animation-delay: .6s; }
.hstat:nth-child(3) .hstat-num { animation-delay: .7s; }
.hstat:nth-child(4) .hstat-num { animation-delay: .8s; }
.hstat-label {
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin-top: 7px;
}

/* ═══════════════════════════════════════════════════════════════════
   PAGE HEADER  (enhanced)
   ═══════════════════════════════════════════════════════════════════ */
.page-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 32px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border);
  animation: fadeDown .5s cubic-bezier(.16,1,.3,1) both;
  position: relative;
}
.page-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 80px; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--pink));
  border-radius: 999px;
  transform-origin: left;
  animation: lineExpand .6s cubic-bezier(.16,1,.3,1) .2s both;
}
.page-icon {
  width: 48px; height: 48px;
  background: linear-gradient(135deg, var(--accent), var(--pink));
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
  flex-shrink: 0;
  box-shadow: 0 8px 24px rgba(108,99,255,.3);
  animation: scaleIn .5s cubic-bezier(.34,1.56,.64,1) .1s both;
}
.page-title {
  font-size: 1.65rem;
  font-weight: 800;
  color: var(--text) !important;
  letter-spacing: -.6px;
  margin: 0;
}
.page-desc {
  font-size: .83rem;
  color: var(--muted) !important;
  margin: 4px 0 0;
}

/* ═══════════════════════════════════════════════════════════════════
   SECTION LABEL
   ═══════════════════════════════════════════════════════════════════ */
.sec-label {
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--accent) !important;
  margin: 32px 0 14px;
  animation: slideRight .4s cubic-bezier(.16,1,.3,1) both;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ═══════════════════════════════════════════════════════════════════
   OVERVIEW STAT STRIP
   ═══════════════════════════════════════════════════════════════════ */
.stat-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
  margin: 24px 0 36px;
}
.scard {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 16px;
  text-align: center;
  animation: cardEntrance .6s cubic-bezier(.16,1,.3,1) both;
  transition: transform .3s cubic-bezier(.34,1.56,.64,1), border-color .25s, box-shadow .25s;
  position: relative;
  overflow: hidden;
  cursor: default;
}
.scard::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% 100%, rgba(61,214,140,.07), transparent 70%);
  opacity: 0;
  transition: opacity .3s;
}
.scard:hover { transform: translateY(-5px) scale(1.02); border-color: var(--green); box-shadow: 0 8px 32px rgba(61,214,140,.15); }
.scard:hover::before { opacity: 1; }
.scard-val {
  font-size: 1.7rem;
  font-weight: 800;
  color: var(--green) !important;
  line-height: 1;
  animation: countUp .5s cubic-bezier(.16,1,.3,1) both;
}
.scard-lbl {
  font-size: .67rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin-top: 7px;
}

/* ═══════════════════════════════════════════════════════════════════
   APPROACH ARCH CARDS
   ═══════════════════════════════════════════════════════════════════ */
.arch-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 18px;
}
.arch-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 26px;
  animation: cardEntrance .6s cubic-bezier(.16,1,.3,1) both;
  transition: transform .3s cubic-bezier(.34,1.56,.64,1), border-color .25s, box-shadow .25s;
  position: relative;
  overflow: hidden;
}
.arch-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% -20%, rgba(108,99,255,.08), transparent 70%);
  opacity: 0;
  transition: opacity .3s;
}
.arch-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  border-radius: 0 0 var(--radius) var(--radius);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform .4s cubic-bezier(.16,1,.3,1);
}
.arch-card.cf::after  { background: linear-gradient(90deg, var(--accent), var(--accentL)); }
.arch-card.cb::after  { background: linear-gradient(90deg, var(--pink), #ff9ed0); }
.arch-card.kb::after  { background: linear-gradient(90deg, var(--green), #8affc8); }
.arch-card:hover::before { opacity: 1; }
.arch-card:hover::after  { transform: scaleX(1); }
.arch-card:hover { transform: translateY(-6px); }
.arch-card.cf:hover { border-color: var(--accent); box-shadow: 0 12px 40px rgba(108,99,255,.18); }
.arch-card.cb:hover { border-color: var(--pink);   box-shadow: 0 12px 40px rgba(255,107,157,.18); }
.arch-card.kb:hover { border-color: var(--green);  box-shadow: 0 12px 40px rgba(61,214,140,.18); }
.arch-card:nth-child(1) { animation-delay: .1s; }
.arch-card:nth-child(2) { animation-delay: .2s; }
.arch-card:nth-child(3) { animation-delay: .3s; }
.arch-icon {
  font-size: 2rem;
  margin-bottom: 12px;
  display: block;
  animation: floatY 3s ease infinite;
}
.arch-card:nth-child(2) .arch-icon { animation-delay: .5s; }
.arch-card:nth-child(3) .arch-icon { animation-delay: 1s; }
.arch-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 14px;
}
.arch-card.cf .arch-title { color: var(--accentL) !important; }
.arch-card.cb .arch-title { color: var(--pink) !important; }
.arch-card.kb .arch-title { color: var(--green) !important; }
.arch-method {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 8px;
  padding: 8px 12px;
  margin: 6px 0;
  font-size: .79rem;
  color: var(--text2) !important;
  font-weight: 500;
  transition: background .2s, border-color .2s, transform .2s;
}
.arch-card:hover .arch-method:hover {
  background: rgba(255,255,255,.08);
  transform: translateX(4px);
}

/* ═══════════════════════════════════════════════════════════════════
   METHOD INFO BANNER
   ═══════════════════════════════════════════════════════════════════ */
.method-banner {
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0;
  font-size: .83rem;
  color: var(--text2) !important;
  line-height: 1.6;
  animation: fadeIn .4s ease both;
  backdrop-filter: blur(4px);
  position: relative;
  overflow: hidden;
}
.method-banner::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.02), transparent);
  animation: shimmer 2s ease infinite;
  background-size: 600px 100%;
}

/* ═══════════════════════════════════════════════════════════════════
   RECOMMENDATION CARDS  (enhanced)
   ═══════════════════════════════════════════════════════════════════ */
.rec-list { margin-top: 18px; }
.rec-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 10px;
  animation: cardEntrance .5s cubic-bezier(.16,1,.3,1) both;
  transition: transform .25s cubic-bezier(.34,1.2,.64,1), border-color .2s, box-shadow .2s;
  display: grid;
  grid-template-columns: 34px 1fr auto;
  gap: 0 14px;
  align-items: start;
  position: relative;
  overflow: hidden;
}
.rec-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--pink));
  transform: scaleY(0);
  transform-origin: top;
  transition: transform .3s cubic-bezier(.16,1,.3,1);
  border-radius: 0 0 0 var(--radius);
}
.rec-card:hover {
  transform: translateX(6px);
  border-color: rgba(108,99,255,.4);
  box-shadow: -4px 0 0 0 var(--accent), 0 6px 28px rgba(108,99,255,.14);
}
.rec-card:hover::before { transform: scaleY(1); }
.rec-rank {
  font-size: .73rem;
  font-weight: 700;
  font-family: var(--mono) !important;
  color: var(--accent) !important;
  background: rgba(108,99,255,.14);
  border: 1px solid rgba(108,99,255,.2);
  border-radius: 8px;
  width: 34px; height: 34px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  transition: background .2s, transform .2s;
}
.rec-card:hover .rec-rank {
  background: rgba(108,99,255,.25);
  transform: scale(1.08) rotate(-3deg);
}
.rec-body { min-width: 0; }
.rec-pid {
  font-size: .87rem;
  font-weight: 700;
  font-family: var(--mono) !important;
  color: var(--text) !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 6px;
}
.rec-bar-bg {
  background: rgba(255,255,255,.06);
  border-radius: 999px;
  height: 5px;
  overflow: hidden;
  margin-bottom: 8px;
}
.rec-bar {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--pink), var(--amber));
  background-size: 200% auto;
  animation: barGrow .8s cubic-bezier(.16,1,.3,1) both, gradientShift 3s ease infinite;
}
.rec-explain {
  font-size: .78rem;
  color: var(--text2) !important;
  line-height: 1.6;
}
.rec-score-badge {
  font-size: .79rem;
  font-weight: 700;
  font-family: var(--mono) !important;
  color: var(--green) !important;
  background: rgba(61,214,140,.1);
  border: 1px solid rgba(61,214,140,.22);
  border-radius: 9px;
  padding: 5px 11px;
  white-space: nowrap;
  flex-shrink: 0;
  align-self: center;
  transition: background .2s, transform .2s;
}
.rec-card:hover .rec-score-badge {
  background: rgba(61,214,140,.18);
  transform: scale(1.05);
}
.rec-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.rec-tag {
  font-size: .67rem;
  font-weight: 600;
  color: var(--muted) !important;
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 5px;
  padding: 2px 8px;
  transition: background .2s;
}
.rec-card:hover .rec-tag {
  background: rgba(255,255,255,.08);
}

/* Staggered card delays */
.rec-card:nth-child(1)  { animation-delay: .04s; }
.rec-card:nth-child(2)  { animation-delay: .09s; }
.rec-card:nth-child(3)  { animation-delay: .14s; }
.rec-card:nth-child(4)  { animation-delay: .19s; }
.rec-card:nth-child(5)  { animation-delay: .24s; }
.rec-card:nth-child(6)  { animation-delay: .29s; }
.rec-card:nth-child(7)  { animation-delay: .34s; }
.rec-card:nth-child(8)  { animation-delay: .39s; }
.rec-card:nth-child(9)  { animation-delay: .44s; }
.rec-card:nth-child(10) { animation-delay: .49s; }

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR OVERRIDES
   ═══════════════════════════════════════════════════════════════════ */
.sidebar-brand {
  text-align: center;
  padding: 22px 0 14px;
}
.sidebar-logo {
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: -1px;
  background: linear-gradient(135deg, var(--accentL), var(--pink));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: floatY 3s ease infinite;
}
.sidebar-sub {
  font-size: .63rem;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin-top: 5px;
}
.sidebar-stats {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 13px 15px;
  margin-top: 10px;
  font-size: .78rem;
  line-height: 1.9;
  animation: fadeUp .4s ease both;
}
.sidebar-stats .lbl { color: var(--muted) !important; }
.sidebar-stats .val { color: var(--text) !important; font-weight: 600; }

/* Streamlit widget fixes */
.stButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, #8b84ff 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: .87rem !important;
  letter-spacing: .04em !important;
  padding: 10px 24px !important;
  transition: opacity .2s, transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .2s !important;
  position: relative;
  overflow: hidden;
}
.stButton > button::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at var(--x,50%) var(--y,50%), rgba(255,255,255,.25), transparent 60%);
  opacity: 0;
  transition: opacity .2s;
}
.stButton > button:hover {
  opacity: .94 !important;
  transform: translateY(-3px) !important;
  box-shadow: 0 8px 28px rgba(108,99,255,.4) !important;
}
.stButton > button:active {
  transform: translateY(-1px) !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stRadio"] label,
div[data-testid="stFileUploader"] label {
  font-size: .76rem !important;
  font-weight: 700 !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input {
  background: var(--card2) !important;
  border-color: var(--border) !important;
  border-radius: 9px !important;
  color: var(--text) !important;
  font-size: .87rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
div[data-testid="stTextInput"] > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(108,99,255,.15) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 10px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
  color: var(--muted) !important;
  font-size: .82rem !important;
}
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--pink)) !important;
  background-size: 200% auto !important;
  border-radius: 999px !important;
  animation: gradientShift 2s ease infinite !important;
}
div[data-testid="stMetricValue"] {
  font-size: 1.6rem !important;
  font-weight: 800 !important;
  color: var(--accentL) !important;
}
.stTabs [data-baseweb="tab"] {
  font-weight: 600 !important;
  font-size: .84rem !important;
  color: var(--muted) !important;
  letter-spacing: .04em !important;
  transition: color .2s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-border"] {
  background: linear-gradient(90deg, var(--accent), var(--pink)) !important;
}
.stDataFrame, .stDataFrame * { font-size: .79rem !important; }
[data-testid="stRadio"] > div { gap: 6px !important; }
[data-testid="stRadio"] > div > label {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
  padding: 8px 14px !important;
  margin: 0 !important;
  font-size: .83rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: border-color .2s, background .2s, transform .15s !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  color: var(--text2) !important;
}
[data-testid="stRadio"] > div > label:hover {
  border-color: rgba(108,99,255,.4) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stRadio"] > div > label:has(input:checked) {
  border-color: var(--accent) !important;
  background: rgba(108,99,255,.12) !important;
  color: var(--text) !important;
  box-shadow: 0 4px 16px rgba(108,99,255,.15) !important;
}
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* Plotly transparent backgrounds */
.js-plotly-plot .plotly,
.js-plotly-plot .plotly div { background: transparent !important; }

/* Staggered scard delays */
.scard:nth-child(1) { animation-delay: .06s; }
.scard:nth-child(2) { animation-delay: .12s; }
.scard:nth-child(3) { animation-delay: .18s; }
.scard:nth-child(4) { animation-delay: .24s; }
.scard:nth-child(5) { animation-delay: .30s; }

/* ── Eval metric tabs animation ──── */
.stTabs { animation: fadeUp .5s cubic-bezier(.16,1,.3,1) both; }

/* ── Info/warning boxes ─────────── */
[data-testid="stInfo"] {
  background: rgba(108,99,255,.08) !important;
  border-left-color: var(--accent) !important;
  border-radius: 10px !important;
  animation: fadeIn .4s ease both;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for k in ["df", "cf", "cb", "kb", "eval_report", "ready"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  (xaxis/yaxis NOT included — passed explicitly per chart)
# ─────────────────────────────────────────────────────────────────────────────

# Base theme WITHOUT xaxis / yaxis so we never get duplicate-kwarg errors
PL_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#b0b0c8", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
)

# Full theme including axes — use when you are NOT passing xaxis/yaxis yourself
PL = dict(
    **PL_BASE,
    xaxis=dict(gridcolor="#2e2e45", linecolor="#2e2e45", tickfont_size=11),
    yaxis=dict(gridcolor="#2e2e45", linecolor="#2e2e45", tickfont_size=11),
)

COLORS = ["#6c63ff", "#ff6b9d", "#3dd68c", "#ffb347", "#64b5f6"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)' — fixes Plotly fillcolor bug."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _esc(text: str) -> str:
    """Escape HTML special characters so dynamic text never breaks markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_rec_cards(recs: list, score_key: str, max_val: float):
    if not recs:
        st.info("No recommendations found — try adjusting your settings.")
        return

    # Render each card as its own st.markdown call.
    # One big HTML blob breaks when explanation text contains quotes / angle-brackets
    # (e.g. the _explain() strings use single-quoted product IDs like 'B008BY7O9W').
    for i, r in enumerate(recs):
        pid   = _esc(r.get("ProductId", "—"))
        score = float(r.get(score_key, r.get("AvgScore", r.get("SimilarityScore", 0))))
        expl  = _esc(r.get("Explanation", ""))
        pct   = round(100 * score / max_val) if max_val else 0
        badge = f"{score:.2f}" if max_val <= 1 else f"★ {score:.2f}"

        tags = []
        if "NumReviews"     in r: tags.append(f"📝 {r['NumReviews']} reviews")
        if "AvgHelpfulness" in r: tags.append(f"👍 {r['AvgHelpfulness']:.2f} helpful")
        tags_html = (
            "<div class='rec-tags'>"
            + "".join(f'<span class="rec-tag">{_esc(t)}</span>' for t in tags)
            + "</div>"
        ) if tags else ""

        st.markdown(f"""
        <div class="rec-card" style="animation-delay:{i*0.05:.2f}s">
          <div class="rec-rank">#{i+1}</div>
          <div class="rec-body">
            <div class="rec-pid">{pid}</div>
            <div class="rec-bar-bg">
              <div class="rec-bar"
                   style="width:{pct}%; --w:{pct}%;
                          animation-delay:{i*0.05+0.15:.2f}s;">
              </div>
            </div>
            <div class="rec-explain">{expl}</div>
            {tags_html}
          </div>
          <div class="rec-score-badge">{badge}</div>
        </div>""", unsafe_allow_html=True)


def page_header(icon: str, title: str, desc: str):
    st.markdown(f"""
    <div class="page-header">
      <div class="page-icon">{icon}</div>
      <div>
        <div class="page-title">{title}</div>
        <div class="page-desc">{desc}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def sec(label: str):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)


def method_banner(method: str, desc: str, color: str):
    st.markdown(f"""
    <div class="method-banner"
         style="background:rgba(255,255,255,.02);
                border:1px solid {color}33;
                border-left:3px solid {color};">
      <strong style="color:{color};">{method}</strong> &mdash; {desc}
    </div>""", unsafe_allow_html=True)


def make_radar(methods, data_dict, title):
    """BUG-FIXED: uses rgba() for fillcolor instead of 8-digit hex."""
    cats = ["Precision", "Recall", "Coverage", "Novelty"]
    fig  = go.Figure()
    for i, m in enumerate(methods):
        v = data_dict.get(m, {})
        pk = next((k for k in v if "Precision" in k), "")
        rk = next((k for k in v if "Recall"    in k), "")
        vals = [
            v.get(pk, 0),
            v.get(rk, 0),
            v.get("Coverage(%)", 0) / 100,
            min(v.get("Novelty", 0) / 15, 1),
        ]
        c    = COLORS[i % len(COLORS)]
        fill = hex_to_rgba(c, 0.14)          # ← FIX: proper rgba string
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            mode="lines+markers", name=m,
            line=dict(color=c, width=2.5),
            marker=dict(size=7, color=c,
                        line=dict(width=1.5, color="#0d0d14")),
            fill="toself",
            fillcolor=fill,
        ))
    fig.update_layout(
        title=dict(text=title, font_size=14, font_color="#f0f0f8"),
        polar=dict(
            bgcolor="rgba(255,255,255,.02)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#2e2e45", linecolor="#2e2e45",
                            tickfont=dict(size=9, color="#6a6a85")),
            angularaxis=dict(gridcolor="#2e2e45", linecolor="#2e2e45",
                             tickfont=dict(size=11, color="#b0b0c8")),
        ),
        legend=dict(font=dict(size=11, color="#b0b0c8"),
                    bgcolor="rgba(0,0,0,0)", bordercolor="#2e2e45"),
        **PL_BASE,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">🧠 RecoSense</div>
      <div class="sidebar-sub">AIE425 · Alamein University</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Load Dataset**")

    local_csv_path = os.path.join(os.path.dirname(__file__), "data", "Reviews.csv")
    file_exists    = os.path.exists(local_csv_path)

    if file_exists:
        st.success("✅ Dataset found (Reviews.csv)")
    else:
        st.error("❌ Please place 'Reviews.csv' inside the 'data' folder.")

    sample_k  = st.slider("Sample size (K rows)", 5, 100, 20, 5,
                          help="20K rows trains in ~15 seconds.")
    build_btn = st.button("⚡  Build Models", use_container_width=True,
                          disabled=not file_exists)

    if st.session_state.ready:
        stat = dataset_stats(st.session_state.df)
        st.markdown(f"""
        <div class="sidebar-stats">
          <div><span class="lbl">Reviews  </span><span class="val">{stat['total_reviews']:,}</span></div>
          <div><span class="lbl">Users    </span><span class="val">{stat['unique_users']:,}</span></div>
          <div><span class="lbl">Products </span><span class="val">{stat['unique_products']:,}</span></div>
          <div><span class="lbl">Avg Score</span><span class="val">{stat['avg_score']}</span></div>
          <div><span class="lbl">Sparsity </span><span class="val">{stat['sparsity']*100:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🧭 Navigation**")
    page = st.radio("nav", [
        "🏠  Overview",
        "🤝  Collaborative Filtering",
        "📄  Content-Based",
        "🧩  Knowledge-Based",
        "📊  Evaluation & Comparison",
    ], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODELS
# ─────────────────────────────────────────────────────────────────────────────

if build_btn and file_exists:
    prog   = st.progress(0, text="Loading data…")
    status = st.empty()
    try:
        prog.progress(10, text="⏳ Reading CSV from disk…")
        df = load_reviews(local_csv_path, sample_size=sample_k * 1000)
        st.session_state.df = df
        prog.progress(30, text=f"✅ Loaded {len(df):,} rows — Training CF (4 models)…")
        st.session_state.cf = CollaborativeFilteringEngine(df)
        prog.progress(60, text="✅ CF done — Building Content-Based profiles…")
        st.session_state.cb = ContentBasedEngine(df)
        prog.progress(80, text="✅ Content-Based done — Building Knowledge catalog…")
        st.session_state.kb = KnowledgeBasedEngine(df)
        prog.progress(95, text="✅ Almost done…")
        st.session_state.eval_report = None
        st.session_state.ready       = True
        prog.progress(100, text="🎉 All models ready!")
        import time; time.sleep(.6)
        prog.empty(); status.empty()
    except Exception as e:
        st.error(f"Error while loading: {e}")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.ready:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
      </div>
      <div class="hero-logo">RecoSense</div>
      <div class="hero-tagline">Intelligent Recommender System</div>
      <div class="hero-pill">AIE425 · Alamein University</div>
      <div class="hero-instruction">
        Place <code>Reviews.csv</code> in the <code>data/</code> folder, set sample size,<br>
        then click <strong>⚡ Build Models</strong> to unlock all engines.
      </div>
      <div class="hero-stats">
        <div class="hstat"><div class="hstat-num">4</div><div class="hstat-label">CF Methods</div></div>
        <div class="hstat"><div class="hstat-num">2</div><div class="hstat-label">CB Methods</div></div>
        <div class="hstat"><div class="hstat-num">3</div><div class="hstat-label">KB Strategies</div></div>
        <div class="hstat"><div class="hstat-num">6</div><div class="hstat-label">Eval Metrics</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# shortcuts
df = st.session_state.df
cf = st.session_state.cf
cb = st.session_state.cb
kb = st.session_state.kb

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if "Overview" in page:
    page_header("🏠", "Overview", "Dataset summary and system architecture")

    stat = dataset_stats(df)
    vals = [f"{stat['total_reviews']:,}", f"{stat['unique_users']:,}",
            f"{stat['unique_products']:,}", str(stat['avg_score']),
            f"{stat['sparsity']*100:.1f}%"]
    lbls = ["Total Reviews", "Unique Users", "Products", "Avg Score", "Sparsity"]
    cards = "".join(
        f'<div class="scard" style="animation-delay:{i*.07:.2f}s">'
        f'<div class="scard-val">{v}</div>'
        f'<div class="scard-lbl">{l}</div></div>'
        for i, (v, l) in enumerate(zip(vals, lbls))
    )
    st.markdown(f'<div class="stat-strip">{cards}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        sec("Score Distribution")
        sd  = stat["score_dist"]
        fig = go.Figure(go.Bar(
            x=[f"★ {k}" for k in sd], y=list(sd.values()),
            marker=dict(color=list(sd.values()),
                        colorscale=[[0,"#2e2e45"],[.5,"#6c63ff"],[1,"#3dd68c"]],
                        line_width=0),
            text=list(sd.values()), textposition="outside",
            textfont=dict(size=11, color="#b0b0c8"),
        ))
        fig.update_layout(title="Reviews per Star Rating",
                          xaxis_title="Score", yaxis_title="Count",
                          showlegend=False, **PL)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Top Active Users")
        top_u = df["UserId"].value_counts().head(10).reset_index()
        top_u.columns = ["UserId", "Reviews"]
        top_u["Short"] = top_u["UserId"].str[:14] + "…"
        fig2 = go.Figure(go.Bar(
            x=top_u["Reviews"], y=top_u["Short"],
            orientation="h",
            marker=dict(color=COLORS[0], line_width=0),
            text=top_u["Reviews"], textposition="outside",
            textfont=dict(size=10, color="#b0b0c8"),
        ))
        fig2.update_layout(title="Most Active Users", yaxis_autorange="reversed",
                           showlegend=False, **PL)
        st.plotly_chart(fig2, use_container_width=True)

    sec("System Architecture")
    st.markdown("""
    <div class="arch-grid">
      <div class="arch-card cf">
        <span class="arch-icon">🤝</span>
        <div class="arch-title">Collaborative Filtering</div>
        <div class="arch-method">① User-Based KNN (cosine)</div>
        <div class="arch-method">② Item-Based KNN (cosine + means)</div>
        <div class="arch-method">③ SVD — Matrix Factorisation</div>
        <div class="arch-method">④ Slope One (deviation-based)</div>
      </div>
      <div class="arch-card cb">
        <span class="arch-icon">📄</span>
        <div class="arch-title">Content-Based</div>
        <div class="arch-method">① TF-IDF + Cosine Similarity</div>
        <div class="arch-method">② Score-Weighted TF-IDF</div>
      </div>
      <div class="arch-card kb">
        <span class="arch-icon">🧩</span>
        <div class="arch-title">Knowledge-Based</div>
        <div class="arch-method">① Score-Ranked</div>
        <div class="arch-method">② Helpfulness-Ranked</div>
        <div class="arch-method">③ Popularity-Ranked</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

elif "Collaborative" in page:
    page_header("🤝", "Collaborative Filtering",
                "Finds users or items with similar rating patterns — 4 methods")

    users  = cf.get_user_list()
    c1, c2, c3 = st.columns([3, 2, 1])
    uid    = c1.selectbox("Select User", users)
    method = c2.selectbox("CF Method", CF_METHODS)
    n_recs = c3.slider("Top-N", 3, 20, 10)

    info = {
        "User-Based KNN":  ("Finds users with similar rating history and borrows their preferences.", COLORS[0]),
        "Item-Based KNN":  ("Finds products frequently rated similarly together.",                    COLORS[1]),
        "SVD":             ("Matrix factorisation: decomposes ratings into latent taste factors.",    COLORS[2]),
        "Slope One":       ("Predicts ratings from the average rating gap between pairs of items.",   COLORS[3]),
    }
    desc, col = info[method]
    method_banner(method, desc, col)

    if st.button("🚀  Get Recommendations", key="cf_go"):
        with st.spinner("Computing…"):
            recs = cf.recommend(uid, method=method, n=n_recs)

        c_left, c_right = st.columns([3, 2])
        with c_left:
            sec(f"Top {n_recs} Recommendations · {method}")
            render_rec_cards(recs, "PredictedScore", 5.0)

        with c_right:
            sec("Score Distribution")
            if recs:
                scores = [r["PredictedScore"] for r in recs]
                pids   = [r["ProductId"][:14] + "…" for r in recs]
                fig = go.Figure(go.Bar(
                    x=scores, y=pids, orientation="h",
                    marker=dict(color=scores,
                                colorscale=[[0,"#2e2e45"],[.5,"#6c63ff"],[1,"#3dd68c"]],
                                line_width=0),
                    text=[f"{s:.2f}" for s in scores], textposition="outside",
                    textfont=dict(size=10),
                ))
                # ── BUG FIX #2: pass xaxis/yaxis explicitly; use PL_BASE ──
                fig.update_layout(
                    title="Predicted Scores",
                    xaxis=dict(range=[0, 5.5],
                               gridcolor="#2e2e45", linecolor="#2e2e45",
                               tickfont_size=11),
                    yaxis=dict(autorange="reversed",
                               gridcolor="#2e2e45", linecolor="#2e2e45",
                               tickfont_size=11),
                    showlegend=False,
                    **PL_BASE,
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CONTENT-BASED
# ─────────────────────────────────────────────────────────────────────────────

elif "Content" in page:
    page_header("📄", "Content-Based Recommendation",
                "Matches products by review text similarity via TF-IDF")

    users  = sorted(df["UserId"].unique().tolist())
    c1, c2, c3 = st.columns([3, 2, 1])
    uid    = c1.selectbox("Select User", users, key="cb_u")
    method = c2.selectbox("CB Method", CB_METHODS, key="cb_m")
    n_recs = c3.slider("Top-N", 3, 20, 10, key="cb_n")

    info_cb = {
        "TF-IDF":          ("Reviews of liked products define your taste; closest unseen products recommended.", COLORS[0]),
        "Weighted-TF-IDF": ("Same as TF-IDF but high-rated products carry more weight in your profile.",       COLORS[1]),
    }
    desc, col = info_cb[method]
    method_banner(method, desc, col)

    if st.button("🚀  Get Recommendations", key="cb_go"):
        with st.spinner("Computing…"):
            recs = cb.recommend(uid, method=method, n=n_recs)

        c_left, c_right = st.columns([3, 2])
        with c_left:
            sec(f"Top {n_recs} · {method}")
            render_rec_cards(recs, "SimilarityScore", 1.0)

        with c_right:
            sec("Similarity vs Community Score")
            if recs:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r["SimilarityScore"]      for r in recs],
                    y=[r["AvgCommunityScore"]     for r in recs],
                    mode="markers+text",
                    text=[r["ProductId"][:10]     for r in recs],
                    textposition="top center",
                    textfont=dict(size=9, color="#b0b0c8"),
                    marker=dict(size=13, color=COLORS[1],
                                line=dict(width=1.5, color="#0d0d14")),
                ))
                fig.update_layout(
                    title="Text Similarity vs Community Rating",
                    xaxis_title="Cosine Similarity",
                    yaxis_title="Avg Community Score",
                    **PL
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KNOWLEDGE-BASED
# ─────────────────────────────────────────────────────────────────────────────

elif "Knowledge" in page:
    page_header("🧩", "Knowledge-Based Recommendation",
                "Filter by constraints — no rating history needed")

    users = sorted(df["UserId"].unique().tolist())
    c1, c2 = st.columns([3, 1])
    uid    = c1.selectbox("Select User (excludes already-rated)", users, key="kb_u")
    n_recs = c2.slider("Top-N", 3, 20, 10, key="kb_n")

    score_lo, score_hi = kb.get_score_range()
    rev_lo,   rev_hi   = kb.get_review_range()

    st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
    sec("Constraints")
    ca, cb2, cc, cd = st.columns(4)
    min_score   = ca.slider("Min avg score ★",  float(score_lo), 5.0, 4.0, 0.1)
    min_reviews = cb2.slider("Min # reviews",    int(rev_lo), min(500, int(rev_hi)), 5)
    max_reviews = cc.slider("Max # reviews",     min_reviews, int(rev_hi), int(rev_hi))
    ranking     = cd.selectbox("Rank by", RANKING_OPTIONS)
    keyword     = st.text_input("Keyword filter (optional)",
                                placeholder="e.g. organic, chocolate, gluten-free")

    if st.button("🔍  Apply & Recommend", key="kb_go"):
        with st.spinner("Filtering catalogue…"):
            recs = kb.recommend(uid, min_score=min_score, keyword=keyword,
                                min_reviews=min_reviews, max_reviews=max_reviews,
                                ranking=ranking, n=n_recs)

        c_left, c_right = st.columns([3, 2])
        with c_left:
            sec(f"Top {n_recs} · {ranking.capitalize()}-ranked")
            render_rec_cards(recs, "AvgScore", 5.0)

        with c_right:
            sec("Score vs Helpfulness")
            if recs:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Avg Score",
                    x=[r["ProductId"][:12]+"…" for r in recs],
                    y=[r["AvgScore"] for r in recs],
                    marker_color=COLORS[0], marker_line_width=0,
                    text=[f"{r['AvgScore']:.2f}" for r in recs], textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    name="Avg Helpfulness",
                    x=[r["ProductId"][:12]+"…" for r in recs],
                    y=[r["AvgHelpfulness"] for r in recs],
                    marker_color=COLORS[2], marker_line_width=0,
                    text=[f"{r['AvgHelpfulness']:.2f}" for r in recs], textposition="outside",
                ))
                fig.update_layout(barmode="group", title="Score & Helpfulness",
                                  showlegend=True, **PL)
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

elif "Evaluation" in page:
    page_header("📊", "Evaluation & Comparison",
                "6 metrics across all approaches and methods")

    n_eval = st.slider("N for Precision@N / Recall@N", 5, 20, 10)

    if st.button("▶  Run Full Evaluation", key="eval_go"):
        with st.spinner("Evaluating all approaches — please wait…"):
            ev     = Evaluator(df, n=n_eval)
            report = ev.run(cf, cb, kb)
            st.session_state.eval_report = report

    report = st.session_state.eval_report
    if report is None:
        st.info("Click **▶ Run Full Evaluation** above to compute all metrics.")
        st.stop()

    # ── RMSE / MAE ──────────────────────────────────────────────────────
    sec("CF Rating Accuracy — RMSE & MAE")
    rmse_rows = [{"Method": m, **v} for m, v in report["CF"]["RMSE_MAE"].items()]
    df_rmse   = pd.DataFrame(rmse_rows)

    c1, c2 = st.columns(2)
    for col, metric, color in [(c1,"RMSE",COLORS[0]),(c2,"MAE",COLORS[1])]:
        fig = go.Figure(go.Bar(
            x=df_rmse["Method"], y=df_rmse[metric],
            marker_color=color, marker_line_width=0,
            text=df_rmse[metric].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig.update_layout(title=f"{metric} by CF Method", showlegend=False, **PL)
        col.plotly_chart(fig, use_container_width=True)

    # ── Radar: CF methods ────────────────────────────────────────────────
    sec("CF Methods — Radar Comparison")
    cf_m = [k for k in report["CF"] if k != "RMSE_MAE"]
    st.plotly_chart(make_radar(cf_m, report["CF"], "CF Methods · Radar"),
                    use_container_width=True)

    # ── Cross-approach tabs ──────────────────────────────────────────────
    sec("All Methods — Cross-Approach Comparison")
    rows = []
    for approach, methods in report.items():
        for mname, vals in methods.items():
            if mname == "RMSE_MAE" or not isinstance(vals, dict): continue
            pk = next((k for k in vals if "Precision" in k), None)
            rk = next((k for k in vals if "Recall"    in k), None)
            rows.append({
                "Approach": approach, "Method": mname,
                "Precision": vals.get(pk, 0),
                "Recall":    vals.get(rk, 0),
                "Coverage":  vals.get("Coverage(%)", 0),
                "Novelty":   vals.get("Novelty", 0),
            })
    df_all = pd.DataFrame(rows)

    tabs = st.tabs(["📍 Precision", "📡 Recall", "🌐 Coverage", "✨ Novelty"])
    for tab, metric in zip(tabs, ["Precision","Recall","Coverage","Novelty"]):
        with tab:
            fig = px.bar(df_all, x="Method", y=metric, color="Approach",
                         color_discrete_sequence=COLORS,
                         title=f"{metric} · All Methods",
                         text_auto=".3f", barmode="group")
            fig.update_layout(showlegend=True, **PL)
            fig.update_traces(marker_line_width=0, textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

    # ── Approach-level radar ─────────────────────────────────────────────
    sec("Approach-Level Radar")
    agg = {}
    for approach, methods in report.items():
        ps, rs, cs, ns = [], [], [], []
        for mname, vals in methods.items():
            if mname == "RMSE_MAE" or not isinstance(vals, dict): continue
            pk = next((k for k in vals if "Precision" in k), None)
            rk = next((k for k in vals if "Recall"    in k), None)
            if pk: ps.append(vals[pk])
            if rk: rs.append(vals[rk])
            cs.append(vals.get("Coverage(%)", 0))
            ns.append(vals.get("Novelty", 0))
        agg[approach] = {
            "Precision@N": np.mean(ps) if ps else 0,
            "Recall@N":    np.mean(rs) if rs else 0,
            "Coverage(%)": np.mean(cs) if cs else 0,
            "Novelty":     np.mean(ns) if ns else 0,
        }
    st.plotly_chart(make_radar(list(agg), agg, "Approach-Level Radar"),
                    use_container_width=True)

    # ── Analysis ─────────────────────────────────────────────────────────
    sec("Analysis & Conclusions")
    st.markdown(Evaluator.analysis_text())

    sec("Full Metrics Table")
    st.dataframe(Evaluator.to_dataframe(report), use_container_width=True)