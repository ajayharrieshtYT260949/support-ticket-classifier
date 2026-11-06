from pathlib import Path
import random
import pandas as pd

RANDOM_SEED = 42
N_PER_CATEGORY_PRIORITY = 30

CATEGORY_TEMPLATES = {
    "Account Access": [
        "I cannot log in to my account because the password is not accepted",
        "The password reset link is not working and I cannot access my profile",
        "My account is locked and I need help signing in",
        "I am unable to verify my email address to access the account",
        "The login page keeps rejecting my credentials",
    ],
    "Billing": [
        "I was charged twice for the same purchase and need a billing correction",
        "The invoice contains an incorrect amount",
        "I do not recognize a charge on my payment statement",
        "My recent payment failed but money was deducted from my bank",
        "Please explain the extra fee on my latest bill",
    ],
    "Technical Issue": [
        "The application crashes whenever I try to open the dashboard",
        "The website displays an error and does not load correctly",
        "A feature stopped working after the latest application update",
        "The page freezes when I submit the form",
        "I am experiencing repeated technical errors while using the service",
    ],
    "Shipping": [
        "My order has not arrived and the tracking page has not updated",
        "The package is delayed beyond the promised delivery date",
        "The tracking number shows no movement for several days",
        "My shipment was delivered to the wrong address",
        "I need an update about the delivery status of my order",
    ],
    "Returns": [
        "I want to return the item and request a refund",
        "The product arrived damaged and I need to arrange a return",
        "Please provide instructions for exchanging the item",
        "My refund has not arrived after the returned product was received",
        "I received the wrong item and would like to send it back",
    ],
    "Product Inquiry": [
        "I would like to know the product specifications before purchasing",
        "Does this product support the features listed on the website",
        "Please share more information about product compatibility",
        "I have a question about the available product options",
        "Can you confirm the dimensions and warranty details",
    ],
    "Subscription": [
        "I need help changing my subscription plan",
        "Please explain how to cancel my recurring subscription",
        "My subscription renewal did not complete successfully",
        "I was moved to a different subscription tier unexpectedly",
        "I want to update the billing cycle for my subscription",
    ],
    "Security": [
        "I noticed a login from an unknown device and suspect unauthorized access",
        "I believe someone has accessed my account without permission",
        "I received a suspicious message asking for my account password",
        "Please help me report a possible security breach",
        "I want to report suspicious activity on my profile",
    ],
}

PRIORITY_PHRASES = {
    "Low": [
        "There is no immediate urgency",
        "This can be handled whenever the team is available",
        "I am contacting support for a routine request",
        "Please review this when convenient",
        "It is not time sensitive",
    ],
    "Medium": [
        "I would appreciate a response soon",
        "This is affecting my normal use of the service",
        "Please look into this during the next business day",
        "I need assistance to resolve this issue",
        "A timely update would be helpful",
    ],
    "High": [
        "This is urgent and needs immediate attention",
        "The issue is blocking my work right now",
        "Please escalate this as soon as possible",
        "I cannot continue until this is resolved",
        "This is causing a serious disruption",
    ],
}

CLOSINGS = [
    "Thank you for your help.",
    "Please let me know the next steps.",
    "I would appreciate your assistance.",
    "I hope this can be resolved soon.",
    "Please contact me with an update.",
    "I have already tried the basic troubleshooting steps.",
    "I can provide more details if required.",
    "I would be grateful for a clear resolution.",
    "Please check this request and advise.",
    "I am waiting for your support team's response.",
    "I have included the relevant details in this request.",
    "Please review the issue and guide me.",
]

def main():
    random.seed(RANDOM_SEED)
    rows = []
    ticket_id = 1

    for category, templates in CATEGORY_TEMPLATES.items():
        for priority, priority_phrases in PRIORITY_PHRASES.items():
            for _ in range(N_PER_CATEGORY_PRIORITY):
                description = " ".join([
                    random.choice(templates),
                    random.choice(priority_phrases),
                    random.choice(CLOSINGS),
                ])
                rows.append({
                    "ticket_id": f"TKT-{ticket_id:05d}",
                    "text": description,
                    "category": category,
                    "priority": priority,
                })
                ticket_id += 1

    random.shuffle(rows)
    output = Path(__file__).resolve().parents[1] / "data" / "raw" / "support_tickets.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print(f"Dataset saved: {output}")
    print(f"Rows: {len(df)}")
    print("\nCategory counts:")
    print(df["category"].value_counts().sort_index())
    print("\nPriority counts:")
    print(df["priority"].value_counts().sort_index())

if __name__ == "__main__":
    main()
