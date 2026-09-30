"""Forensic-examination-bench dashboard for the eval-agent judge.

Design language: a forensic lab bench, not a sci-fi HUD. Warm paper,
ink-black hairlines, one signal-red accent spent only on failures, sharp
corners, monospace ledgers, rubber-stamp verdicts. (The neon-glass look is
what every generated dashboard looks like; this one has a point of view.)

Run:
    uvicorn dashboard:app --reload --port 8000   # http://localhost:8000

Demo mode: if GROQ_API_KEY is not set, judge endpoints return a canned demo
verdict badged as SPECIMEN so the UI is explorable without a key.
"""

import json
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from eval_agent.judge import JudgeAgent
from eval_agent.schemas import (
    ClaimVerdict,
    ComparisonVerdict,
    HallucinationResult,
    MetricScore,
    UsageStats,
    Verdict,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_JSON = BASE_DIR / "reports" / "results.json"
DEMO_MODE = not os.environ.get("GROQ_API_KEY")
JUDGE_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="eval-agent // forensic bench")


class JudgeRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    sources: List[str] = Field(min_length=1)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    answer_a: str = Field(min_length=1)
    answer_b: str = Field(min_length=1)
    sources: List[str] = Field(min_length=1)


_judge: JudgeAgent | None = None


def get_judge() -> JudgeAgent:
    global _judge
    if _judge is None:
        _judge = JudgeAgent()
    return _judge


def _demo_verdict() -> Verdict:
    def m(score: int, rationale: str) -> MetricScore:
        label = "pass" if score >= 4 else "borderline" if score == 3 else "fail"
        return MetricScore(score=score, verdict=label, rationale=rationale)  # type: ignore[arg-type]

    hall = HallucinationResult(
        **m(2, "The answer invents a cascading database failure that the incident report never mentions.").model_dump(),
        unsupported_spans=["a cascading database failure that corrupted several records"],
    )
    claims = [
        ClaimVerdict(text="The May 3rd outage lasted 2 hours", status="supported",
                     evidence="a 2-hour outage"),
        ClaimVerdict(text="Caused by a misconfigured load balancer during a routine deploy",
                     status="supported", evidence="caused by a misconfigured load balancer during a routine deploy"),
        ClaimVerdict(text="No customer data was lost", status="supported",
                     evidence="No customer data was lost."),
        ClaimVerdict(text="Root cause was a cascading database failure that corrupted several records",
                     status="contradicted", evidence="No customer data was lost."),
    ]
    return Verdict.from_metrics(
        faithfulness=m(2, "Three claims check out, but the invented database failure contradicts the source."),
        relevance=m(5, "Directly answers what caused the outage; nothing off-topic."),
        hallucination=hall,
        claims=claims,
        summary="A fluent answer smuggling in one fabricated root cause. The invented "
                "\"cascading database failure\" fails it on faithfulness.",
        model=JUDGE_MODEL,
        usage=UsageStats(prompt_tokens=1240, completion_tokens=312, total_tokens=1552, latency_ms=1834),
        case_id="adv-04",
        demo=True,
    )


def _demo_comparison() -> ComparisonVerdict:
    return ComparisonVerdict(
        winner="B", score_a=2, score_b=5,
        rationale="Answer B restates the incident report exactly: 2-hour outage, misconfigured load balancer, "
                  "no data lost. Answer A adds a cascading database failure the report never mentions, "
                  "contradicting its own 'no data lost' claim.",
        deciding_factor="Answer A invents a root cause; answer B invents nothing.",
        usage=UsageStats(prompt_tokens=1510, completion_tokens=204, total_tokens=1714, latency_ms=2210),
        model=JUDGE_MODEL,
        demo=True,
    )


@app.post("/api/judge")
def api_judge(req: JudgeRequest):
    if DEMO_MODE:
        return JSONResponse(_demo_verdict().model_dump())
    return JSONResponse(get_judge().judge(req.question, req.answer, req.sources).model_dump())


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    if DEMO_MODE:
        return JSONResponse(_demo_comparison().model_dump())
    return JSONResponse(
        get_judge().compare(req.question, req.answer_a, req.answer_b, req.sources).model_dump()
    )


@app.get("/api/sample")
def api_sample():
    cases = json.loads((BASE_DIR / "data" / "test_cases.json").read_text(encoding="utf-8"))
    trap = next(c for c in cases if c["id"] == "adv-04")
    return {"question": trap["question"], "answer": trap["answer"], "sources": trap["sources"]}


@app.get("/api/traps")
def api_traps():
    cases = json.loads((BASE_DIR / "data" / "test_cases.json").read_text(encoding="utf-8"))
    return [
        {"id": c["id"], "trap": c["notes"], "question": c["question"],
         "answer": c["answer"], "sources": c["sources"]}
        for c in cases if c["category"] == "adversarial"
    ]


@app.get("/api/report")
def api_report():
    if not REPORT_JSON.exists():
        return {"available": False}
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return {"available": True, "model": data["model"], "aggregate": data["aggregate"]}


@app.get("/api/status")
def api_status():
    return {"demo": DEMO_MODE, "model": JUDGE_MODEL}


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVAL-AGENT — Forensic Eval Bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#f3efe3; --paper2:#eae5d3; --ink:#191510; --ink60:rgba(25,21,16,.6);
    --ink15:rgba(25,21,16,.14); --ink08:rgba(25,21,16,.08);
    --red:#c22418; --redink:#a01c12; --amber:#9a6a00;
  }
  *{box-sizing:border-box;border-radius:0!important}
  html,body{margin:0;padding:0}
  body{
    background-color:var(--paper); color:var(--ink);
    background-image:
      linear-gradient(var(--ink08) 1px,transparent 1px),
      linear-gradient(90deg,var(--ink08) 1px,transparent 1px),
      linear-gradient(rgba(25,21,16,.13) 1px,transparent 1px),
      linear-gradient(90deg,rgba(25,21,16,.13) 1px,transparent 1px);
    background-size:26px 26px,26px 26px,130px 130px,130px 130px;
    font-family:"Archivo",system-ui,sans-serif; min-height:100vh;
  }
  ::selection{background:var(--ink);color:var(--paper)}
  .wrap{max-width:1240px;margin:0 auto;padding:0 26px 70px}
  /* ---------- ticker ---------- */
  .ticker{background:var(--ink);color:var(--paper);overflow:hidden;white-space:nowrap;
    border-bottom:3px solid var(--ink)}
  .ticker-inner{display:inline-block;padding:9px 0;font-family:"JetBrains Mono",monospace;
    font-size:12px;letter-spacing:.28em;animation:tick 26s linear infinite}
  @keyframes tick{to{transform:translateX(-50%)}}
  .ticker b{color:var(--red);font-weight:700}
  /* ---------- header ---------- */
  header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;
    padding:34px 0 6px;flex-wrap:wrap}
  .brand h1{font-weight:900;font-size:clamp(44px,7vw,84px);letter-spacing:-.03em;
    margin:0;line-height:.95}
  .brand h1 .thin{font-weight:500}
  .brand .rule{height:6px;background:var(--ink);margin:12px 0 10px;position:relative}
  .brand .rule::after{content:"";position:absolute;right:0;top:0;width:34%;height:100%;background:var(--red)}
  .brand p{font-family:"JetBrains Mono",monospace;font-size:12px;letter-spacing:.24em;margin:0;color:var(--ink60)}
  .meta{text-align:right;font-family:"JetBrains Mono",monospace;font-size:12px;line-height:2;letter-spacing:.08em}
  .meta .tag{display:inline-block;border:2px solid var(--ink);padding:3px 12px;font-weight:700;
    letter-spacing:.2em;font-size:11px;margin-top:6px}
  .meta .tag.demo{border-color:var(--red);color:var(--red);transform:rotate(-3deg)}
  .meta .tag.live{background:var(--ink);color:var(--paper)}
  /* ---------- tabs ---------- */
  .tabs{display:flex;margin:26px 0 0;border:2px solid var(--ink);background:var(--paper)}
  .tab{flex:1;padding:15px 10px;background:transparent;border:none;border-right:2px solid var(--ink);
    font-family:"JetBrains Mono",monospace;font-size:13px;font-weight:700;letter-spacing:.18em;
    cursor:pointer;color:var(--ink);transition:background .12s,color .12s}
  .tab:last-child{border-right:none}
  .tab:hover{background:var(--ink15)}
  .tab.active{background:var(--ink);color:var(--paper)}
  .tab .n{color:var(--red);margin-right:8px}
  .tab.active .n{color:var(--paper)}
  /* ---------- panels ---------- */
  .grid{display:grid;grid-template-columns:1fr 1.15fr;gap:0;margin-top:26px;border:2px solid var(--ink)}
  @media(max-width:960px){.grid{grid-template-columns:1fr}.grid>.panel+.panel{border-left:none;border-top:2px solid var(--ink)}}
  .grid>.panel+.panel{border-left:2px solid var(--ink)}
  .panel{background:rgba(243,239,227,.88);padding:26px 28px;position:relative;min-width:0}
  .panel.shake{animation:shake .4s}
  @keyframes shake{20%{transform:translate(-4px,2px)}40%{transform:translate(4px,-2px)}60%{transform:translate(-3px,-2px)}80%{transform:translate(2px,2px)}}
  .sec-head{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid var(--ink);
    padding-bottom:10px;margin-bottom:20px}
  .sec-head .idx{font-family:"JetBrains Mono",monospace;font-weight:700;font-size:13px;
    background:var(--ink);color:var(--paper);padding:3px 9px;letter-spacing:.1em}
  .sec-head h2{margin:0;font-size:15px;font-weight:800;letter-spacing:.22em}
  .sec-head .hint{margin-left:auto;font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--ink60);letter-spacing:.08em}
  label{display:block;font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;
    letter-spacing:.22em;margin:18px 0 7px}
  label .arw{color:var(--red)}
  textarea{width:100%;background:#fbf9f1;border:2px solid var(--ink);color:var(--ink);
    padding:12px 14px;font-size:14px;resize:vertical;font-family:"Archivo",sans-serif;line-height:1.55}
  textarea.mono{font-family:"JetBrains Mono",monospace;font-size:13px}
  textarea:focus{outline:3px solid var(--red);outline-offset:-3px}
  .src-row{display:flex;gap:0;margin-bottom:10px}
  .src-row textarea{flex:1;border-right:none}
  .src-row .docn{font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;background:var(--ink);
    color:var(--paper);display:flex;align-items:center;padding:0 12px;letter-spacing:.1em}
  .icon-btn{background:#fbf9f1;border:2px solid var(--ink);color:var(--red);min-width:46px;
    cursor:pointer;font-size:17px;font-weight:700}
  .icon-btn:hover{background:var(--red);color:var(--paper);border-color:var(--red)}
  .btn-row{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap}
  button.primary{flex:1;min-width:220px;padding:16px;border:2px solid var(--ink);cursor:pointer;
    font-family:"JetBrains Mono",monospace;font-size:15px;font-weight:700;letter-spacing:.24em;
    background:var(--red);color:#fff;box-shadow:5px 5px 0 var(--ink);
    transition:transform .08s,box-shadow .08s;position:relative;overflow:hidden}
  button.primary:hover{transform:translate(2px,2px);box-shadow:3px 3px 0 var(--ink)}
  button.primary:active{transform:translate(5px,5px);box-shadow:0 0 0 var(--ink)}
  button.primary:disabled{opacity:.6;cursor:wait;transform:none}
  button.primary .scan{position:absolute;top:0;bottom:0;width:70px;left:-90px;display:none;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent)}
  button.primary.busy .scan{display:block;animation:scan 1s linear infinite}
  @keyframes scan{to{left:110%}}
  button.ghost{padding:13px 18px;border:2px solid var(--ink);cursor:pointer;background:#fbf9f1;
    font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:700;letter-spacing:.14em;
    box-shadow:4px 4px 0 var(--ink);transition:transform .08s,box-shadow .08s}
  button.ghost:hover{transform:translate(2px,2px);box-shadow:2px 2px 0 var(--ink)}
  button.ghost:active{transform:translate(4px,4px);box-shadow:0 0 0 var(--ink)}
  /* ---------- verdict: the stamp ---------- */
  .empty{padding:90px 20px;text-align:center;font-family:"JetBrains Mono",monospace;
    font-size:13px;letter-spacing:.1em;color:var(--ink60);line-height:2}
  .empty .big{font-size:52px;display:block;letter-spacing:0}
  .stamp-zone{text-align:center;padding:26px 0 8px;position:relative}
  .stamp{display:inline-block;border:5px double var(--ink);padding:12px 34px;
    font-weight:900;font-size:40px;letter-spacing:.2em;color:var(--ink);
    transform:rotate(-7deg);background:rgba(251,249,241,.6)}
  .stamp.fail{border-color:var(--red);color:var(--red)}
  .stamp.borderline{border-color:var(--amber);color:var(--amber)}
  .stamp.slam{animation:slam .5s cubic-bezier(.2,2.2,.4,1) both}
  @keyframes slam{0%{transform:scale(2.6) rotate(-16deg);opacity:0}
    55%{transform:scale(.93) rotate(-6deg);opacity:1}100%{transform:scale(1) rotate(-7deg)}}
  .score-line{font-family:"JetBrains Mono",monospace;font-size:15px;font-weight:700;
    letter-spacing:.3em;margin-top:14px}
  .score-line .fail{color:var(--red)} .score-line .pass{color:var(--ink)} .score-line .borderline{color:var(--amber)}
  /* ---------- instrument meters ---------- */
  .meters{margin:24px 0 6px;border-top:2px solid var(--ink)}
  .meter{padding:13px 0;border-bottom:1px dashed var(--ink60)}
  .meter-head{display:flex;justify-content:space-between;font-family:"JetBrains Mono",monospace;
    font-size:12px;font-weight:700;letter-spacing:.18em;margin-bottom:8px}
  .meter-head .bad{color:var(--red)} .meter-head .ok{color:var(--ink)} .meter-head .warn{color:var(--amber)}
  .meter-track{height:16px;border:2px solid var(--ink);background:#fbf9f1;position:relative}
  .meter-fill{height:100%;width:0;background:var(--ink)}
  .meter.fail .meter-fill{background:var(--red)}
  .meter.borderline .meter-fill{background:var(--amber)}
  .meter.go .meter-fill{animation:fillbar .9s steps(14) forwards}
  @keyframes fillbar{to{width:var(--w)}}
  /* ---------- ledger stats ---------- */
  .ledger{margin:20px 0;border:2px solid var(--ink)}
  .ledger .row{display:flex;justify-content:space-between;padding:9px 14px;
    font-family:"JetBrains Mono",monospace;font-size:12px;letter-spacing:.1em;border-bottom:1px solid var(--ink15)}
  .ledger .row:last-child{border-bottom:none}
  .ledger .row:nth-child(odd){background:var(--ink08)}
  .ledger .k{color:var(--ink60)} .ledger .v{font-weight:700}
  /* ---------- metric blocks ---------- */
  .metric{border:2px solid var(--ink);margin:14px 0;background:#fbf9f1;animation:rise .4s both}
  .metric.fail{border-color:var(--red)}
  .metric .mhead{display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;border-bottom:2px solid var(--ink);font-family:"JetBrains Mono",monospace;
    font-size:12px;font-weight:700;letter-spacing:.2em}
  .metric.fail .mhead{border-color:var(--red);background:var(--red);color:#fff}
  .metric p{margin:0;padding:12px 14px;font-size:14px;line-height:1.6}
  @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  /* ---------- marker highlights + claim ledger ---------- */
  .span-chip{display:inline-block;margin:6px 8px 0 0;padding:7px 12px;font-family:"JetBrains Mono",monospace;
    font-size:13px;background:rgba(194,36,24,.18);border:2px solid var(--red);color:var(--redink);
    font-weight:700;animation:rise .4s both;transform:rotate(-.6deg)}
  table.claims{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:4px}
  table.claims th{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.2em;text-align:left;
    padding:9px 10px;border-bottom:2px solid var(--ink);color:var(--ink60)}
  table.claims td{padding:11px 10px;border-bottom:1px solid var(--ink15);vertical-align:top;line-height:1.55}
  table.claims tr:last-child td{border-bottom:none}
  table.claims tr.reveal{animation:printrow .35s both}
  @keyframes printrow{from{opacity:0;background:var(--ink);color:var(--paper)}to{opacity:1}}
  .mk{font-family:"JetBrains Mono",monospace;font-weight:700;font-size:11px;letter-spacing:.1em;
    border:2px solid var(--ink);padding:3px 8px;white-space:nowrap;display:inline-block}
  .mk.ok{background:var(--ink);color:var(--paper)}
  .mk.no{background:var(--red);border-color:var(--red);color:#fff}
  .mk.maybe{background:transparent;color:var(--amber);border-color:var(--amber)}
  td.ev{font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--ink60)}
  td.ev.red{color:var(--redink);font-weight:700}
  .note{margin-top:18px;border:2px solid var(--ink);background:#fbf9f1;position:relative}
  .note .nhead{font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;letter-spacing:.24em;
    background:var(--ink);color:var(--paper);padding:8px 14px}
  .note .nbody{padding:14px;font-size:14.5px;line-height:1.7;min-height:76px}
  .note .nbody .cur{display:inline-block;width:10px;height:18px;background:var(--red);vertical-align:-3px;animation:blink .7s steps(1) infinite}
  @keyframes blink{50%{opacity:0}}
  .specimen{margin-top:14px;border:2px dashed var(--red);color:var(--redink);padding:10px 14px;
    font-family:"JetBrains Mono",monospace;font-size:12px;letter-spacing:.06em;transform:rotate(-.8deg);
    background:rgba(194,36,24,.06)}
  .examining{font-family:"JetBrains Mono",monospace;letter-spacing:.2em;font-size:13px;
    padding:60px 20px;text-align:center}
  .examining .dots::after{content:"";animation:dots 1.2s steps(4) infinite}
  @keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
  /* ---------- compare ---------- */
  .duel{display:grid;grid-template-columns:1fr 1fr;gap:0;border:2px solid var(--ink);margin:20px 0}
  @media(max-width:700px){.duel{grid-template-columns:1fr}}
  .side{padding:22px;text-align:center;position:relative;background:rgba(243,239,227,.88)}
  .side+.side{border-left:2px solid var(--ink)}
  .side.lose{opacity:.45;filter:grayscale(.6)}
  .side .nm{font-weight:900;font-size:30px;letter-spacing:.1em}
  .side .sc{font-family:"JetBrains Mono",monospace;font-size:44px;font-weight:700;margin:6px 0}
  .side .st{font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;letter-spacing:.24em}
  .side.win .st{color:var(--red)}
  .crown{position:absolute;top:-24px;left:50%;transform:translateX(-50%) rotate(-6deg);
    background:var(--ink);color:var(--paper);font-family:"JetBrains Mono",monospace;font-size:11px;
    font-weight:700;letter-spacing:.2em;padding:6px 14px;animation:rise .4s both}
  .vsline{text-align:center;font-weight:900;font-size:22px;letter-spacing:.4em;margin:2px 0;color:var(--red)}
  /* ---------- gallery ---------- */
  .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:0;
    border:2px solid var(--ink);margin-top:26px}
  .trap{background:rgba(243,239,227,.88);border-right:2px solid var(--ink);border-bottom:2px solid var(--ink);
    padding:22px;cursor:pointer;position:relative;transition:background .12s}
  .trap:hover{background:var(--ink);color:var(--paper)}
  .trap:hover .tdesc{color:rgba(243,239,227,.75)}
  .trap:hover .go{color:var(--red)}
  .trap .tno{font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;letter-spacing:.24em;color:var(--ink60)}
  .trap:hover .tno{color:rgba(243,239,227,.6)}
  .trap h3{margin:10px 0 8px;font-size:19px;font-weight:800;letter-spacing:-.01em}
  .trap .tdesc{font-size:13.5px;line-height:1.6;color:var(--ink60)}
  .trap .tflag{position:absolute;top:14px;right:14px;background:var(--red);color:#fff;
    font-family:"JetBrains Mono",monospace;font-size:10px;font-weight:700;letter-spacing:.2em;padding:4px 10px;
    transform:rotate(4deg)}
  .trap .go{margin-top:14px;font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:700;letter-spacing:.2em}
  /* ---------- report ---------- */
  .report{margin-top:26px;border:2px solid var(--ink);background:rgba(243,239,227,.88)}
  table.rep{width:100%;border-collapse:collapse;font-size:13px}
  table.rep th{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.2em;text-align:left;
    padding:10px 14px;border-bottom:2px solid var(--ink);color:var(--ink60)}
  table.rep td{padding:9px 14px;border-bottom:1px solid var(--ink15);font-family:"JetBrains Mono",monospace}
  footer{margin-top:40px;border-top:3px solid var(--ink);padding-top:14px;display:flex;
    justify-content:space-between;font-family:"JetBrains Mono",monospace;font-size:11px;
    letter-spacing:.2em;color:var(--ink60);flex-wrap:wrap;gap:8px}
  .err{padding:30px;text-align:center;color:var(--red);font-family:"JetBrains Mono",monospace;font-weight:700}
  .tabpage{display:none}.tabpage.active{display:block}
</style>
</head>
<body>
<div class="ticker"><div class="ticker-inner" id="ticker"></div></div>
<div class="wrap">
  <header>
    <div class="brand">
      <h1>EVAL<span class="thin">—</span>AGENT</h1>
      <div class="rule"></div>
      <p>LLM-AS-JUDGE // FORENSIC EVAL BENCH</p>
    </div>
    <div class="meta">
      JUDGE&nbsp;·&nbsp;<span id="modelLine">…</span><br>
      BENCH&nbsp;Nº&nbsp;001 — HALLUCINATION UNIT<br>
      <span class="tag demo" id="modeBadge" style="display:none">SPECIMEN · DEMO</span>
      <span class="tag live" id="liveBadge" style="display:none">● LIVE</span>
    </div>
  </header>

  <nav class="tabs">
    <button class="tab active" data-tab="judge" onclick="switchTab('judge')"><span class="n">01</span>SINGLE JUDGE</button>
    <button class="tab" data-tab="compare" onclick="switchTab('compare')"><span class="n">02</span>A/B COMPARE</button>
    <button class="tab" data-tab="gallery" onclick="switchTab('gallery')"><span class="n">03</span>TRAP GALLERY</button>
  </nav>

  <!-- 01 SINGLE JUDGE -->
  <div class="tabpage active" id="tab-judge">
    <div class="grid">
      <div class="panel">
        <div class="sec-head"><span class="idx">01</span><h2>CASE FILE</h2><span class="hint">EXHIBIT A</span></div>
        <label>QUESTION <span class="arw">▸</span></label>
        <textarea id="q" rows="2" placeholder="What the user asked…"></textarea>
        <label>ANSWER UNDER EXAMINATION <span class="arw">▸</span></label>
        <textarea id="a" rows="5" placeholder="The chatbot / RAG answer…"></textarea>
        <label>SOURCE DOCUMENTS <span class="arw">▸</span></label>
        <div id="sources"></div>
        <div class="btn-row">
          <button class="ghost" onclick="addSource('')">+ ADD SOURCE</button>
          <button class="ghost" onclick="loadSample()">⚡ LOAD TRAP CASE</button>
        </div>
        <div class="btn-row"><button class="primary" id="judgeBtn" onclick="runJudge()"><span class="scan"></span><span id="judgeLbl">▶&nbsp; EXAMINE</span></button></div>
      </div>
      <div class="panel" id="resultPanel">
        <div class="sec-head"><span class="idx">02</span><h2>FINDINGS</h2><span class="hint">EXAMINER'S REPORT</span></div>
        <div class="empty" id="emptyState"><span class="big">⚖</span>NO EXAMINATION IN PROGRESS.<br>FILE A CASE — OR LOAD THE TRAP.</div>
        <div id="verdictBody" style="display:none"></div>
      </div>
    </div>
  </div>

  <!-- 02 A/B COMPARE -->
  <div class="tabpage" id="tab-compare">
    <div class="grid">
      <div class="panel">
        <div class="sec-head"><span class="idx">01</span><h2>TWO CONTENDERS</h2><span class="hint">EXHIBIT B/C</span></div>
        <label>QUESTION <span class="arw">▸</span></label>
        <textarea id="cq" rows="2" placeholder="What the user asked…"></textarea>
        <label>ANSWER A <span class="arw">▸</span></label>
        <textarea id="ca" rows="4" placeholder="First candidate…"></textarea>
        <label>ANSWER B <span class="arw">▸</span></label>
        <textarea id="cb" rows="4" placeholder="Second candidate…"></textarea>
        <label>SOURCE DOCUMENTS <span class="arw">▸</span></label>
        <div id="csources"></div>
        <div class="btn-row"><button class="ghost" onclick="addCSource('')">+ ADD SOURCE</button></div>
        <div class="btn-row"><button class="primary" id="cmpBtn" onclick="runCompare()"><span class="scan"></span><span id="cmpLbl">⚔&nbsp; PIT THEM</span></button></div>
      </div>
      <div class="panel">
        <div class="sec-head"><span class="idx">02</span><h2>THE DECISION</h2><span class="hint">ONE SURVIVES</span></div>
        <div class="empty" id="cmpEmpty"><span class="big">⚔</span>TWO ANSWERS ENTER.<br>ONE VERDICT LEAVES.</div>
        <div id="cmpBody" style="display:none"></div>
      </div>
    </div>
  </div>

  <!-- 03 TRAP GALLERY -->
  <div class="tabpage" id="tab-gallery">
    <div class="gallery" id="gallery"></div>
  </div>

  <div class="report">
    <div class="sec-head" style="margin:0;padding:16px 20px"><span class="idx">04</span><h2>BATCH LEDGER</h2><span class="hint">LATEST TEST-SET RUN</span></div>
    <div id="reportBody" class="empty" style="padding:34px">NO BATCH RUN ON RECORD — <span style="font-weight:700">python -m eval_agent.runner</span></div>
  </div>

  <footer><span>EVAL-AGENT · GROQ-POWERED LLM JUDGE</span><span>SET IN ARCHIVO + JETBRAINS MONO</span><span>BUILT BY IBRAHIM HAMID</span></footer>
</div>

<script>
/* ticker */
const tickTxt='EVAL-AGENT <b>▸</b> FAITHFULNESS <b>▸</b> RELEVANCE <b>▸</b> HALLUCINATION DETECTION <b>▸</b> CLAIM-LEVEL AUDIT <b>▸</b> ';
document.getElementById('ticker').innerHTML=(tickTxt.repeat(6));

function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  document.querySelectorAll('.tabpage').forEach(p=>p.classList.toggle('active',p.id==='tab-'+name));
}
const qp=new URLSearchParams(location.search);
if(qp.get('tab')==='compare')switchTab('compare');
if(qp.get('tab')==='gallery')switchTab('gallery');

function addSource(v,n){const d=document.createElement('div');d.className='src-row';
  d.innerHTML=`<span class="docn">D1</span><textarea class="mono" rows="3" placeholder="Paste source document…"></textarea><button class="icon-btn" onclick="this.parentElement.remove()">✕</button>`;
  d.querySelector('textarea').value=v||'';document.getElementById('sources').appendChild(d);
  document.querySelectorAll('#sources .docn').forEach((el,i)=>el.textContent='D'+(i+1));}
function addCSource(v){const d=document.createElement('div');d.className='src-row';
  d.innerHTML=`<textarea class="mono" rows="2" placeholder="Paste source document…"></textarea><button class="icon-btn" onclick="this.parentElement.remove()">✕</button>`;
  d.querySelector('textarea').value=v||'';document.getElementById('csources').appendChild(d);}
addSource('');addSource('');addCSource('');

async function loadSample(){const r=await fetch('/api/sample');const s=await r.json();
  document.getElementById('q').value=s.question;document.getElementById('a').value=s.answer;
  document.getElementById('sources').innerHTML='';s.sources.forEach(v=>addSource(v));}

function meter(label,metric,i){
  const pct=Math.round(metric.score/5*100);
  const cls=metric.verdict==='fail'?'fail':metric.verdict==='borderline'?'borderline':'';
  const tag=metric.verdict==='fail'?'bad':metric.verdict==='borderline'?'warn':'ok';
  return `<div class="meter ${cls}" style="animation:rise .4s ${i*.08}s both">
    <div class="meter-head"><span>${label}</span><span class="${tag}">${metric.score}/5 — ${metric.verdict.toUpperCase()}</span></div>
    <div class="meter-track"><div class="meter-fill" style="--w:${pct}%"></div></div></div>`;
}
function startMeters(scope){requestAnimationFrame(()=>requestAnimationFrame(()=>
  scope.querySelectorAll('.meter').forEach(m=>m.classList.add('go'))));}

let lastVerdict=null;
async function runJudge(auto){
  const q=document.getElementById('q').value.trim(),a=document.getElementById('a').value.trim();
  const sources=[...document.querySelectorAll('#sources textarea')].map(t=>t.value.trim()).filter(Boolean);
  if(!q||!a||!sources.length){if(!auto)alert('File the question, the answer, and at least one source.');return;}
  const btn=document.getElementById('judgeBtn'),lbl=document.getElementById('judgeLbl');
  btn.disabled=true;btn.classList.add('busy');lbl.innerHTML='EXAMINING…';
  document.getElementById('emptyState').style.display='none';
  const body=document.getElementById('verdictBody');body.style.display='block';
  body.innerHTML='<div class="examining">CROSS-REFERENCING CLAIMS AGAINST SOURCES<span class="dots"></span></div>';
  await new Promise(r=>setTimeout(r,900)); /* purposeful beat: let the examination feel real */
  try{
    const r=await fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,answer:a,sources})});
    if(!r.ok)throw new Error(await r.text());
    const v=await r.json();lastVerdict=v;
    const mk=s=>s==='supported'?'<span class="mk ok">[OK]</span>':s==='contradicted'?'<span class="mk no">[X]</span>':'<span class="mk maybe">[?]</span>';
    body.innerHTML=`
      <div class="stamp-zone"><div class="stamp ${v.overall_verdict} slam">${v.overall_verdict.toUpperCase()}</div>
        <div class="score-line">COMPOSITE&nbsp;<span class="${v.overall_verdict}" id="scoreBig">0.00</span>&nbsp;/&nbsp;5.00</div></div>
      <div class="meters">${meter('FAITHFULNESS',v.faithfulness,0)}${meter('RELEVANCE',v.relevance,1)}${meter('HALLUCINATION',v.hallucination,2)}</div>
      <div class="ledger">
        <div class="row"><span class="k">EXAMINATION TIME</span><span class="v">${v.usage.latency_ms} MS</span></div>
        <div class="row"><span class="k">TOKENS CONSUMED</span><span class="v">${v.usage.total_tokens}</span></div>
        <div class="row"><span class="k">CLAIMS AUDITED</span><span class="v">${v.claims.length}</span></div>
        <div class="row"><span class="k">JUDGE MODEL</span><span class="v">${v.model}</span></div>
      </div>
      <div class="metric ${v.faithfulness.verdict==='fail'?'fail':''}" style="animation-delay:.1s">
        <div class="mhead"><span>FAITHFULNESS</span><span>${v.faithfulness.score}/5 · ${v.faithfulness.verdict.toUpperCase()}</span></div><p>${v.faithfulness.rationale}</p></div>
      <div class="metric" style="animation-delay:.16s">
        <div class="mhead"><span>RELEVANCE</span><span>${v.relevance.score}/5 · ${v.relevance.verdict.toUpperCase()}</span></div><p>${v.relevance.rationale}</p></div>
      <div class="metric ${v.hallucination.verdict==='fail'?'fail':''}" style="animation-delay:.22s">
        <div class="mhead"><span>HALLUCINATION</span><span>${v.hallucination.score}/5 · ${v.hallucination.verdict.toUpperCase()}</span></div><p>${v.hallucination.rationale}</p>
        ${v.hallucination.unsupported_spans.length?`<div style="padding:0 14px 14px"><div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.2em;margin-bottom:4px">MARKED EXHIBITS — UNSUPPORTED SPANS</div>${v.hallucination.unsupported_spans.map((s,i)=>`<span class="span-chip" style="animation-delay:${.3+i*.08}s">“${s}”</span>`).join('')}</div>`:''}</div>
      ${v.claims.length?`<div class="sec-head" style="margin-top:22px"><span class="idx">§</span><h2>CLAIM LEDGER</h2><span class="hint">${v.claims.length} ENTRIES</span></div>
        <table class="claims"><tr><th>#</th><th>CLAIM</th><th>STATUS</th><th>SOURCE EVIDENCE</th></tr>
        ${v.claims.map((c,i)=>`<tr class="reveal" style="animation-delay:${.15+i*.12}s"><td style="font-family:'JetBrains Mono',monospace;color:var(--ink60)">${String(i+1).padStart(2,'0')}</td><td>${c.text}</td><td>${mk(c.status)}</td><td class="ev ${c.status==='contradicted'?'red':''}">${c.evidence?('“'+c.evidence+'”'):'—'}</td></tr>`).join('')}</table>`:''}
      <div class="note"><div class="nhead">EXAMINER'S NOTE</div><div class="nbody" id="noteBody"></div></div>
      <div class="btn-row"><button class="ghost" onclick="exportVerdict()">⬇ EXPORT FINDINGS (JSON)</button></div>
      ${v.demo?`<div class="specimen">SPECIMEN — canned demo findings for bench preview. Set GROQ_API_KEY and restart for live examinations.</div>`:''}`;
    const panel=document.getElementById('resultPanel');panel.classList.remove('shake');void panel.offsetWidth;panel.classList.add('shake');
    startMeters(body);
    const el=document.getElementById('scoreBig'),t0=performance.now();
    (function cnt(t){const p=Math.min(1,(t-t0)/800);el.textContent=(v.overall_score*(1-Math.pow(1-p,3))).toFixed(2);if(p<1)requestAnimationFrame(cnt);})(t0);
    const nb=document.getElementById('noteBody');let i=0;
    const tw=setInterval(()=>{nb.innerHTML=v.summary.slice(0,++i)+'<span class="cur"></span>';if(i>=v.summary.length)clearInterval(tw);},13);
  }catch(e){body.innerHTML=`<div class="err">EXAMINATION ABORTED — ${String(e).slice(0,200)}</div>`;}
  btn.disabled=false;btn.classList.remove('busy');lbl.innerHTML='▶&nbsp; EXAMINE';
}
function exportVerdict(){if(!lastVerdict)return;
  const b=new Blob([JSON.stringify(lastVerdict,null,2)],{type:'application/json'});
  const u=URL.createObjectURL(b),l=document.createElement('a');
  l.href=u;l.download='findings.json';l.click();URL.revokeObjectURL(u);}

async function runCompare(auto){
  const q=document.getElementById('cq').value.trim(),a=document.getElementById('ca').value.trim(),b=document.getElementById('cb').value.trim();
  const sources=[...document.querySelectorAll('#csources textarea')].map(t=>t.value.trim()).filter(Boolean);
  if(!q||!a||!b||!sources.length){if(!auto)alert('File the question, both answers, and at least one source.');return;}
  const btn=document.getElementById('cmpBtn'),lbl=document.getElementById('cmpLbl');
  btn.disabled=true;btn.classList.add('busy');lbl.textContent='WEIGHING…';
  document.getElementById('cmpEmpty').style.display='none';
  const body=document.getElementById('cmpBody');body.style.display='block';
  body.innerHTML='<div class="examining">WEIGHING BOTH ANSWERS<span class="dots"></span></div>';
  await new Promise(r=>setTimeout(r,900));
  try{
    const r=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,answer_a:a,answer_b:b,sources})});
    if(!r.ok)throw new Error(await r.text());
    const v=await r.json();
    const side=(nm,sc,win,lose)=>`<div class="side ${win?'win':''} ${lose?'lose':''}">
      ${win?'<div class="crown">★ UPHELD</div>':''}
      <div class="nm">${nm}</div><div class="sc">${sc}<span style="font-size:18px;color:var(--ink60)">/5</span></div>
      <div class="st">${win?'PREVAILS':lose?'DISMISSED':'DEAD HEAT'}</div></div>`;
    body.innerHTML=`
      <div class="stamp-zone"><div class="stamp ${v.winner==='tie'?'borderline':''} slam" style="font-size:30px">${v.winner==='tie'?'DEAD HEAT':'ANSWER '+v.winner+' UPHELD'}</div></div>
      <div class="vsline">— VS —</div>
      <div class="duel">${side('A',v.score_a,v.winner==='A',v.winner==='B')}${side('B',v.score_b,v.winner==='B',v.winner==='A')}</div>
      <div class="metric" style="animation-delay:.15s"><div class="mhead"><span>WHY</span></div><p>${v.rationale}</p></div>
      <div class="metric" style="animation-delay:.22s"><div class="mhead"><span>DECIDING FACTOR</span></div><p>${v.deciding_factor}</p></div>
      <div class="ledger">
        <div class="row"><span class="k">EXAMINATION TIME</span><span class="v">${v.usage.latency_ms} MS</span></div>
        <div class="row"><span class="k">TOKENS CONSUMED</span><span class="v">${v.usage.total_tokens}</span></div>
      </div>
      ${v.demo?`<div class="specimen">SPECIMEN — canned demo decision. Set GROQ_API_KEY and restart for live examinations.</div>`:''}`;
  }catch(e){body.innerHTML=`<div class="err">EXAMINATION ABORTED — ${String(e).slice(0,200)}</div>`;}
  btn.disabled=false;btn.classList.remove('busy');lbl.innerHTML='⚔&nbsp; PIT THEM';
}

async function loadGallery(){
  const r=await fetch('/api/traps');const traps=await r.json();
  const titles={'adv-01':'The Plausible Beta','adv-02':'Multi-Hop Fusion','adv-03':'Branch Mix-Up',
    'adv-04':'The Phantom Root Cause','adv-05':'Stale vs Fresh','adv-06':'Off-By-One Warranty'};
  document.getElementById('gallery').innerHTML=traps.map(t=>`
    <div class="trap" onclick='judgeTrap(${JSON.stringify(t.id)})'>
      <span class="tflag">TRAP</span>
      <div class="tno">FILE ${t.id.toUpperCase()}</div><h3>${titles[t.id]||t.id}</h3>
      <div class="tdesc">${t.trap}</div><div class="go">OPEN FILE →</div></div>`).join('');
  window._traps=Object.fromEntries(traps.map(t=>[t.id,t]));
}
function judgeTrap(id){const t=window._traps[id];if(!t)return;
  document.getElementById('q').value=t.question;document.getElementById('a').value=t.answer;
  document.getElementById('sources').innerHTML='';t.sources.forEach(v=>addSource(v));
  switchTab('judge');runJudge(true);}

async function loadStatus(){const r=await fetch('/api/status');const s=await r.json();
  document.getElementById('modelLine').textContent=s.model.toUpperCase();
  document.getElementById(s.demo?'modeBadge':'liveBadge').style.display='inline-block';}
async function loadReport(){
  try{const r=await fetch('/api/report');const d=await r.json();if(!d.available)return;
    const a=d.aggregate;let h=`<table class="rep"><tr><th>METRIC</th><th>PASS</th><th>BORDERLINE</th><th>FAIL</th><th>PASS RATE</th></tr>`;
    for(const m of ['faithfulness','relevance','hallucination','overall']){const x=a.by_metric[m];
      h+=`<tr><td>${m.toUpperCase()}</td><td>${x.pass}</td><td>${x.borderline}</td><td>${x.fail}</td><td>${(x.pass_rate*100).toFixed(1)}%</td></tr>`;}
    h+='</table><table class="rep"><tr><th>CATEGORY</th><th>CASES</th><th>PASS</th><th>FAIL</th><th>PASS RATE</th></tr>';
    for(const [c,x] of Object.entries(a.by_category)){
      h+=`<tr><td>${c.toUpperCase()}</td><td>${x.total}</td><td>${x.pass}</td><td>${x.fail}</td><td>${(x.pass_rate*100).toFixed(1)}%</td></tr>`;}
    document.getElementById('reportBody').innerHTML=h+'</table>';
  }catch(e){}
}
loadStatus();loadGallery();loadReport();
if(qp.get('demo')==='1'){loadSample().then(()=>runJudge(true));}
if(qp.get('democmp')==='1'){(async()=>{const s=await (await fetch('/api/sample')).json();
  switchTab('compare');
  document.getElementById('cq').value=s.question;document.getElementById('ca').value=s.answer;
  document.getElementById('cb').value='The May 3rd outage lasted 2 hours and was caused by a misconfigured load balancer during a routine deploy. No customer data was lost.';
  document.getElementById('csources').innerHTML='';s.sources.forEach(addCSource);runCompare(true);})();}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
