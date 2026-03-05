"""Email delivery via Resend or SMTP fallback."""

from __future__ import annotations

import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from hashlib import md5
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.config import Settings
from src.storage.models import Article, ArticleSummary, Digest, DigestArticle, User
from src.utils import mask_email as _mask_email
from src.web.digest_token import sign_digest_click, sign_unsubscribe

logger = logging.getLogger(__name__)


WITTY_REMARKS = [
    "Today's models are yesterday's baselines.",
    "Another day, another architecture nobody asked for.",
    "The only thing scaling faster than LLMs is the number of LLM wrappers.",
    "Remember when 'attention' just meant paying it?",
    "Your daily dose of gradient descent into the AI rabbit hole.",
    "Still waiting for AGI, but here's what happened while we wait.",
    "More parameters, more problems.",
    "The singularity is near, but so is lunch.",
    "Transformers: more than meets the API.",
    "If you can't beat the benchmark, change the benchmark.",
    "Today's SOTA is tomorrow's baseline.",
    "Brought to you by the power of matrix multiplication.",
    "Less hype, more signal. That's the deal.",
    "The best time to read about AI was yesterday. The second best time is now.",
    "Warning: may contain traces of actual insight.",
    "No hallucinations here. Probably.",
    "Curated with care, ranked with math.",
    "Your competitive advantage, delivered daily.",
    "Because scrolling Twitter for AI news is so 2023.",
    "Hand-picked by algorithms, reviewed by nobody.",
    "Today's breakthroughs, tomorrow's pip install.",
    "We read the papers so you don't have to. (You should still read them.)",
    "Freshly scraped, ethically sourced AI news.",
    "Where every day is a new epoch.",
    "Keep calm and fine-tune on.",
    "All signal, no noise. Well, mostly.",
    "Your gradient has been updated.",
    "Inference time: now.",
    "The weights have been adjusted in your favor.",
    "One newsletter to rule them all, and in the embedding space bind them.",
]

TEMPLATE_DIR = Path(__file__).parent / "templates"
MAX_SEND_RETRIES = 3


class EmailSender:
    """Multi-provider email sender with digest rendering and retry logic.

    Supports Resend API, Gmail SMTP, generic SMTP, and console (dev) output.
    """

    def __init__(self, settings: Settings):
        self.provider = settings.email.provider
        self.from_address = settings.email.from_address
        self.base_url = getattr(settings, "base_url", "") or ""
        self.secret_key = settings.secret_key
        self.resend_api_key = settings.resend_api_key
        self.smtp_host = settings.email.smtp_host
        self.smtp_port = settings.email.smtp_port
        self.smtp_username = settings.email.smtp_username
        self.smtp_password = settings.email.smtp_password

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=True,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Test SMTP connection. Returns (success, detail_message)."""
        if self.provider == "console":
            return True, "Console provider: no connection needed"
        if self.provider == "resend":
            return True, "Resend provider: connection tested on first send"
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            return True, f"SMTP connection to {self.smtp_host}:{self.smtp_port} succeeded"
        except Exception as e:
            return False, f"SMTP connection failed: {e}"

    def send_verification_email(self, user, verification_url: str) -> bool:
        template = self.jinja_env.get_template("verify_email.html")
        html_body = template.render(user_name=user.name, verification_url=verification_url)
        return self.send(user.email, "Verify your email — The AI Signal", html_body)

    def send_password_reset_email(self, user, reset_url: str) -> bool:
        template = self.jinja_env.get_template("reset_password_email.html")
        html_body = template.render(user_name=user.name, reset_url=reset_url)
        return self.send(user.email, "Reset your password — The AI Signal", html_body)

    def render_digest(
        self,
        digest: Digest,
        articles: list[dict],
        user: User,
        research_articles: list[dict] | None = None,
        explore_articles: list[dict] | None = None,
    ) -> str:
        template = self.jinja_env.get_template("digest.html")
        base_url = self.base_url.rstrip("/") if self.base_url else "http://localhost:8000"

        # Deterministic daily remark: hash date + user_id for stable but unique pick
        date_str = digest.digest_date.strftime("%Y-%m-%d")
        seed = int(md5(f"{date_str}-{user.id}".encode()).hexdigest(), 16)
        witty_remark = WITTY_REMARKS[seed % len(WITTY_REMARKS)]

        def click_url(article_id: int, section: str = "main") -> str:
            token = sign_digest_click(
                self.secret_key, user.id, article_id, digest.id, section,
            )
            return f"{base_url}/api/digest/click?t={token}"

        unsub_token = sign_unsubscribe(self.secret_key, user.id, user.email)
        unsub_url = f"{base_url}/unsubscribe?t={unsub_token}"

        return template.render(
            digest=digest,
            articles=articles,
            research_articles=research_articles or [],
            explore_articles=explore_articles or [],
            user=user,
            date=digest.digest_date.strftime("%B %d, %Y"),
            base_url=base_url,
            witty_remark=witty_remark,
            click_url=click_url,
            unsub_url=unsub_url,
        )

    def send(self, to_email: str, subject: str, html_body: str) -> bool:
        """Send email with retry logic. Returns True on success."""
        if '\n' in to_email or '\r' in to_email or ',' in to_email or ';' in to_email:
            logger.error(f"Rejected suspicious to_email for {_mask_email(to_email)}")
            return False

        if self.provider == "console":
            return self._send_console(to_email, subject, html_body)

        for attempt in range(1, MAX_SEND_RETRIES + 1):
            if self.provider == "resend":
                ok = self._send_resend(to_email, subject, html_body)
            elif self.provider == "gmail":
                ok = self._send_gmail(to_email, subject, html_body)
            else:
                ok = self._send_smtp(to_email, subject, html_body)
            if ok:
                return True
            logger.warning(f"Email send attempt {attempt}/{MAX_SEND_RETRIES} failed for {_mask_email(to_email)}")
            if attempt < MAX_SEND_RETRIES:
                time.sleep(2 ** attempt)

        logger.error(f"All {MAX_SEND_RETRIES} email send attempts failed for {_mask_email(to_email)}")
        logger.error("Falling back to console output for debugging")
        self._send_console(to_email, subject, html_body)
        return False

    def _send_console(self, to_email: str, subject: str, html_body: str) -> bool:
        """Log email to console (dev mode). Extracts URLs for easy clicking."""
        import re
        urls = re.findall(r'href="(http[^"]+)"', html_body)
        logger.info(f"[CONSOLE EMAIL] To: {_mask_email(to_email)} | Subject: {subject}")
        for url in urls:
            logger.debug(f"[CONSOLE EMAIL] Link: {url}")
        return True

    def _send_resend(self, to_email: str, subject: str, html_body: str) -> bool:
        try:
            import resend
            resend.api_key = self.resend_api_key
            resend.Emails.send({
                "from": self.from_address,
                "to": to_email,
                "subject": subject,
                "html": html_body,
            })
            logger.info(f"Email sent to {_mask_email(to_email)} via Resend")
            return True
        except Exception as e:
            logger.error(f"Resend send failed: {e}")
            return False

    def _send_gmail(self, to_email: str, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = to_email
            msg.attach(MIMEText(html_body, "html"))

            _, sender_email = parseaddr(self.from_address)
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(sender_email, to_email, msg.as_string())
            logger.info(f"Email sent to {_mask_email(to_email)} via Gmail SMTP")
            return True
        except Exception as e:
            logger.error(f"Gmail SMTP send failed: {e}")
            return False

    def _send_smtp(self, to_email: str, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = to_email
            msg.attach(MIMEText(html_body, "html"))

            _, sender_email = parseaddr(self.from_address)
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.sendmail(sender_email, to_email, msg.as_string())
            logger.info(f"Email sent to {_mask_email(to_email)} via SMTP")
            return True
        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            return False
