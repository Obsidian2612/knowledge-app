"""
Categories and their fixed starter questions.
Edit this freely — add/remove categories or questions any time; re-run
`python -m app.seed_data` and it will only add what's missing.
"""

CATEGORIES = [
    {
        "slug": "farming",
        "name": "Farming",
        "description": "Crop and livestock knowledge, seasonal timing, soil, equipment.",
        "starters": [
            "What crops or livestock do you have the most experience with?",
            "What's a mistake you see beginners make constantly?",
            "How do you decide when it's time to plant or harvest?",
            "What's a rule of thumb you use that isn't written in any manual?",
        ],
    },
    {
        "slug": "auto-mechanics",
        "name": "Auto Mechanics",
        "description": "Diagnostics, repair procedures, tools, common failures.",
        "starters": [
            "What's a hard-to-diagnose problem you've solved, and how did you figure it out?",
            "What tools do you consider non-negotiable for your work?",
            "What's a common repair that's often done wrong, and what's the right way?",
            "What symptoms reliably point to a specific underlying cause, in your experience?",
        ],
    },
    {
        "slug": "irrigation-engineering",
        "name": "Irrigation Engineering",
        "description": "System design, water management, pumps, scheduling.",
        "starters": [
            "How do you size an irrigation system for a new site?",
            "What's the most common design mistake you encounter?",
            "How do you decide between drip, sprinkler, or flood irrigation for a given situation?",
            "What do you check first when a system underperforms?",
        ],
    },
    {
        "slug": "music-production",
        "name": "Music Production",
        "description": "Recording, mixing, mastering, sound design, gear.",
        "starters": [
            "What's your go-to workflow from raw recording to finished mix?",
            "What mixing mistake do you hear most often in amateur tracks?",
            "What's a piece of gear or plugin you rely on, and why?",
            "How do you approach a genre or sound you haven't worked with before?",
        ],
    },
    {
        "slug": "electrical",
        "name": "Electrical",
        "description": "Wiring, code compliance, troubleshooting, safety.",
        "starters": [
            "What's a wiring mistake you see often that's actually dangerous?",
            "How do you systematically troubleshoot a circuit that isn't working?",
            "What code requirements do people most often get wrong?",
            "What's a judgment call you make on the job that isn't spelled out in any code book?",
        ],
    },
    {
        "slug": "microcontroller-automation",
        "name": "Microcontroller Automation",
        "description": "Embedded systems, sensors, control logic, firmware.",
        "starters": [
            "What platforms or chips do you use most, and why those?",
            "What's a debugging technique you rely on for flaky hardware behavior?",
            "How do you approach power management in a battery-powered design?",
            "What's a design pattern you reuse across projects?",
        ],
    },
    {
        "slug": "iot",
        "name": "IoT",
        "description": "Connectivity, protocols, device management, architecture.",
        "starters": [
            "How do you decide which protocol (MQTT, HTTP, Zigbee, LoRa, etc.) fits a given project?",
            "What's your approach to handling unreliable connectivity in the field?",
            "What security practices do you consider essential for IoT deployments?",
            "What's a scaling problem you've hit going from prototype to many devices?",
        ],
    },
    {
        "slug": "plumbing",
        "name": "Plumbing",
        "description": "Installation, repair, code, diagnostics.",
        "starters": [
            "What's a plumbing problem that's often misdiagnosed?",
            "What's your process for tracing a leak you can't see?",
            "What installation shortcuts cause problems down the line?",
            "What's a rule of thumb you use that a novice wouldn't know?",
        ],
    },
    {
        "slug": "enduser-computing",
        "name": "End User Computing",
        "description": "Desktop support, device management, troubleshooting.",
        "starters": [
            "What's the most common issue you troubleshoot, and your fastest path to a fix?",
            "What's your approach to rolling out changes without disrupting users?",
            "What's a tool or script that saves you the most time?",
            "What's a support ticket pattern that usually signals a deeper problem?",
        ],
    },
    {
        "slug": "sql-server-management",
        "name": "SQL Server Management",
        "description": "Administration, performance tuning, backups, security.",
        "starters": [
            "What's your go-to process for diagnosing a slow query or server?",
            "What backup/recovery strategy do you use, and why?",
            "What configuration mistakes cause the most pain later?",
            "What do you check first when something's wrong and you don't know what yet?",
        ],
    },
    {
        "slug": "civil-engineering",
        "name": "Civil Engineering",
        "description": "Structures, site work, materials, standards.",
        "starters": [
            "What's a design assumption that frequently gets challenged in the field?",
            "What site conditions change your approach the most?",
            "What's a failure mode you've seen that a design review should have caught?",
            "What rule of thumb do you use for quick sanity-checks on a design?",
        ],
    },
    {
        "slug": "ohs-standards",
        "name": "Occupational Health & Safety Standards",
        "description": "Compliance, risk assessment, incident prevention.",
        "starters": [
            "What's a hazard that's commonly underestimated on site?",
            "How do you run a risk assessment for a new or unusual task?",
            "What's a compliance requirement that's frequently misunderstood?",
            "What's a near-miss you've seen that changed how you approach safety?",
        ],
    },
]


def run():
    from app.database import SessionLocal, engine, Base
    from app.models import Category, StarterQuestion

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for cat in CATEGORIES:
            existing = db.query(Category).filter_by(slug=cat["slug"]).first()
            if existing:
                continue
            c = Category(slug=cat["slug"], name=cat["name"], description=cat["description"])
            db.add(c)
            db.flush()
            for i, q in enumerate(cat["starters"]):
                db.add(StarterQuestion(category_id=c.id, order=i, text=q))
            c.current_question = cat["starters"][0]
            print(f"Seeded: {cat['name']}")
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    run()
