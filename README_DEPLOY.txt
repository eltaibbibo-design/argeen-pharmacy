Argeen Pharmacy - Public Deployment

Files:
- app.py
- medicines_intents_ready.json
- Argeen_pharmacy_intents.json
- requirements.txt
- Procfile
- templates/index.html
- static/argeen-logo.png

Local run:
  pip install -r requirements.txt
  python app.py

Production:
  Build command: pip install -r requirements.txt
  Start command: gunicorn app:app

Required environment variables on the hosting provider:
  AI_ENABLED=true
  OPENAI_API_KEY=YOUR_REAL_KEY
  OPENAI_MODEL=gpt-5.6-luna

IMPORTANT:
- Never upload a real OPENAI_API_KEY to GitHub.
- The pharmacy/local medicine logic runs first.
- OpenAI is used as a fallback for general/external questions when the local system has no answer.
- Greetings, thanks, farewells, pharmacy location/hours/services, and medicine-data questions are handled locally.
