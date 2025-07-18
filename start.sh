#!/bin/bash

# Start Rasa in the background with CORS enabled
rasa run --enable-api --cors "*" &

# Wait for a few seconds to let Rasa initialize (optional but safer)
sleep 10

# Start Flask App
python app.py
