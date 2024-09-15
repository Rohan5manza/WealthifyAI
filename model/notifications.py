import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

def send_email_notification(email: str, message: str):
    try:
        msg = Mail(
            from_email='your-email@example.com',
            to_emails=email,
            subject='Stock Price Alert',
            plain_text_content=message
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(msg)
        return response
    except Exception as e:
        print(f"Error sending email: {e}")
        return None
