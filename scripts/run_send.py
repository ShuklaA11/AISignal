#!/usr/bin/env python3
"""Manual trigger: build and send personalized digests for all active users."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings
from src.email_delivery.sender import EmailSender
from src.personalization.digest_builder import build_digest_for_user
from src.storage.database import get_session, init_db
from src.storage.models import ArticleSummary, DigestArticle
from src.storage.queries import get_active_users
from sqlmodel import select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    settings = load_settings()
    init_db(settings.database_url)
    session = get_session(settings.database_url)
    sender = EmailSender(settings)

    try:
        users = get_active_users(session)
        if not users:
            print("No active users found.")
            return

        for user in users:
            # Build personalized digest
            digest = build_digest_for_user(session, user)

            # Get digest articles with summaries
            stmt = (
                select(DigestArticle)
                .where(DigestArticle.digest_id == digest.id)
                .order_by(DigestArticle.display_order)
            )
            links = list(session.exec(stmt).all())

            articles_data = []
            for link in links:
                article = link.article
                # Get role-specific summary
                summary_stmt = (
                    select(ArticleSummary)
                    .where(ArticleSummary.article_id == article.id)
                    .where(ArticleSummary.role == user.role)
                )
                summary = session.exec(summary_stmt).first()

                articles_data.append({
                    "title": article.title,
                    "url": article.url,
                    "source_name": article.source_name,
                    "summary": summary.summary_text if summary else None,
                    "topics": article.topics,
                    "score": link.personalized_score,
                })

            # Render and send
            html = sender.render_digest(digest, articles_data, user)
            success = sender.send(
                to_email=user.email,
                subject=digest.subject_line or f"AI Digest",
                html_body=html,
            )

            if success:
                digest.status = "sent"
                from datetime import datetime
                digest.sent_at = datetime.utcnow()
                session.add(digest)
                session.commit()
                print(f"Sent digest to {user.email} ({len(articles_data)} articles)")
            else:
                print(f"Failed to send digest to {user.email}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
