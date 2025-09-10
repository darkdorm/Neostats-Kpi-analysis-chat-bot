# utils/pdf_export.py
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from datetime import datetime

def export_pdf(kpis, insights, chat_history, filename="kpi_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Title
    flow.append(Paragraph("📊 KPI Assistant Report", styles['Title']))
    flow.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    flow.append(Spacer(1, 20))

    # KPI Section
    flow.append(Paragraph("Key KPIs", styles['Heading2']))
    for col, info in kpis.items():
        growth = f"{(info['growth_fraction']*100):.2f}%" if info['growth_fraction'] else "N/A"
        flow.append(Paragraph(f"<b>{col}</b>: Last={info['last']}, Total={info['total']:.2f}, Growth={growth}", styles['Normal']))
    flow.append(Spacer(1, 20))

    # Insights Section
    flow.append(Paragraph("Auto Insights", styles['Heading2']))
    flow.append(Paragraph(insights if insights else "No insights generated.", styles['Normal']))
    flow.append(Spacer(1, 20))

    # Chat History Section
    flow.append(Paragraph("Chat History", styles['Heading2']))
    for msg in chat_history:
        role = msg['role'].capitalize()
        flow.append(Paragraph(f"<b>{role}:</b> {msg['content']}", styles['Normal']))
        flow.append(Spacer(1, 5))

    doc.build(flow)
    return filename
