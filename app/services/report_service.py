# app/services/report_service.py
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from typing import List, Dict

def make_csv_report(candidate_id: str, events: List[Dict]) -> bytes:
    df = pd.DataFrame(events)
    return df.to_csv(index=False).encode("utf-8")

def make_pdf_report(candidate_id: str, events: List[Dict], summary: Dict, metadata: Dict = None) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    W, H = letter
    y = H - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Proctoring Report - Candidate {candidate_id}")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(60, y, f"Integrity Score: {summary.get('score')}")
    y -= 12
    c.drawString(60, y, f"Total Deductions: {summary.get('total_deductions')}")
    y -= 14

    c.drawString(60, y, "Event counts:")
    y -= 12
    for k, v in (summary.get("counts") or {}).items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 12
        if y < 80:
            c.showPage()
            y = H - 50

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Events (last 50):")
    y -= 16
    c.setFont("Helvetica", 9)
    for e in events[-50:]:
        ts = e.get("timestamp") or e.get("frame_time") or ""
        s = f"{ts} | {e.get('event_type')} | {e.get('extra', '')}"
        if y < 60:
            c.showPage()
            y = H - 50
        c.drawString(50, y, s[:150])
        y -= 12

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
