"""Email delivery of the revision infographic as a PDF attachment.

Primary transport: Resend (https://resend.com) over HTTPS - works on Replit,
where outbound SMTP ports may be blocked. SMTP is kept as a fallback for other
hosts. Configuration is read from environment variables so no credentials live
in code:

    RESEND_API_KEY   Resend API key (enables the feature)
    RESEND_FROM      From address, e.g. "QUB Finance AI Tutor <tutor@yourdomain.ac.uk>".
                     The domain must be verified in Resend. Defaults to
                     "QUB Finance AI Tutor <onboarding@resend.dev>", which Resend only
                     delivers to the account owner's own address (fine for testing).

    SMTP_HOST / SMTP_PORT / SMTP_USERNAME / SMTP_PASSWORD / SMTP_FROM / SMTP_FROM_NAME
                     Fallback SMTP transport, used only when RESEND_API_KEY is unset.

If neither is configured, `is_email_configured()` is False and the option is
hidden in the UI.
"""

import base64
import io
import json
import logging
import os
import re
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from email.utils import formataddr

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

FROM_NAME_DEFAULT = "QUB Finance AI Tutor"
RESEND_API_URL = "https://api.resend.com/emails"
RESEND_FROM_DEFAULT = f"{FROM_NAME_DEFAULT} <onboarding@resend.dev>"
EMAIL_SUBJECT = "Your revision infographic (PDF)"
USER_AGENT = "QUB-Finance-AI-Tutor/1.0 (Flask; +https://github.com/garethfcampbell/StudyApp)"


def _resend_configured():
    return bool(os.getenv("RESEND_API_KEY", "").strip())


def _smtp_configured():
    return bool(os.getenv("SMTP_HOST", "").strip())


def is_email_configured():
    return _resend_configured() or _smtp_configured()


def email_transport():
    """Name of the transport that will be used, for logging."""
    if _resend_configured():
        return "resend"
    if _smtp_configured():
        return "smtp"
    return None


def normalise_email(address):
    """Return the trimmed, lower-cased address, or None if it is not a plausible email."""
    if not address or not isinstance(address, str):
        return None
    address = address.strip()
    if len(address) > 254 or not _EMAIL_RE.match(address):
        return None
    return address.lower()


def build_infographic_pdf(image_b64, title="Revision infographic"):
    """Wrap the PNG infographic in a single-page PDF sized to the image's aspect
    ratio (A4 width). Returns the PDF bytes."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    image_bytes = base64.b64decode(image_b64)
    reader = ImageReader(io.BytesIO(image_bytes))
    img_w, img_h = reader.getSize()

    page_w = 595.27  # A4 width in points
    page_h = page_w * img_h / img_w

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(page_w, page_h))
    pdf.setTitle(title)
    pdf.setAuthor(FROM_NAME_DEFAULT)
    pdf.drawImage(reader, 0, 0, width=page_w, height=page_h, preserveAspectRatio=True)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _email_body(document_name=None, from_name=FROM_NAME_DEFAULT):
    doc_line = f" for \"{document_name}\"" if document_name else ""
    return (
        "Hello,\n\n"
        f"Attached is the one-page revision infographic{doc_line} you requested from the "
        "QUB Finance AI Tutor.\n\n"
        "It summarises the key concepts and general formulas from your uploaded material. "
        "It is a revision aid generated with AI - always check it against your lecture "
        "material, and never present AI output as your own work.\n\n"
        "Good luck with your revision!\n"
        f"{from_name}\n"
    )


# --------------------------------------------------------------------------- Resend
def _send_via_resend(to_email, pdf_bytes, filename, document_name=None):
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    from_addr = os.getenv("RESEND_FROM", "").strip() or RESEND_FROM_DEFAULT
    logging.info(f"INFOGRAPHIC EMAIL: calling Resend API (from={from_addr}, key=...{api_key[-4:]})")
    payload = {
        "from": from_addr,
        "to": [to_email],
        "subject": EMAIL_SUBJECT,
        "text": _email_body(document_name),
        "attachments": [{
            "filename": filename,
            "content": base64.b64encode(pdf_bytes).decode("ascii"),
        }],
    }
    req = urllib.request.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Resend's API is fronted by Cloudflare, which rejects Python's
            # default "Python-urllib/x.y" agent with error 1010 (bot signature).
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Resend rejected the email (HTTP {e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Resend: {e.reason}") from e
    if status not in (200, 201):
        raise RuntimeError(f"Resend returned HTTP {status}: {body[:300]}")
    try:
        message_id = json.loads(body).get("id")
    except Exception:
        message_id = None
    logging.info(f"INFOGRAPHIC EMAIL: Resend accepted PDF ({len(pdf_bytes)} bytes), id={message_id}")


# --------------------------------------------------------------------------- SMTP fallback
def _send_via_smtp(to_email, pdf_bytes, filename, document_name=None):
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    from_addr = os.getenv("SMTP_FROM", "").strip() or username
    if not from_addr:
        raise RuntimeError("Email delivery is not configured (SMTP_FROM / SMTP_USERNAME unset)")
    from_name = os.getenv("SMTP_FROM_NAME", FROM_NAME_DEFAULT)

    msg = EmailMessage()
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = formataddr((from_name, from_addr))
    msg["To"] = to_email
    msg.set_content(_email_body(document_name, from_name))
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    context = ssl.create_default_context()
    timeout = 30
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=timeout) as server:
            if username:
                server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if username:
                server.login(username, password)
            server.send_message(msg)
    logging.info(f"INFOGRAPHIC EMAIL: SMTP sent PDF ({len(pdf_bytes)} bytes)")


def send_infographic_email(to_email, pdf_bytes, filename="revision-infographic.pdf",
                           document_name=None):
    """Send the PDF as an attachment via Resend (preferred) or SMTP. Raises on failure."""
    if _resend_configured():
        return _send_via_resend(to_email, pdf_bytes, filename, document_name)
    if _smtp_configured():
        return _send_via_smtp(to_email, pdf_bytes, filename, document_name)
    raise RuntimeError("Email delivery is not configured (set RESEND_API_KEY, or SMTP_HOST)")


def email_infographic(to_email, image_b64, document_name=None):
    """Build the PDF and send it. Raises on failure."""
    pdf_bytes = build_infographic_pdf(image_b64)
    send_infographic_email(to_email, pdf_bytes, document_name=document_name)
