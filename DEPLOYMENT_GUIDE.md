# Deployment Guide

This guide takes you from zero to a **live, public web app on Hugging Face
Spaces**, plus how to get the free API keys that unlock the full LLM agent.

Everything here is free. The app already runs without any keys (demo mode); keys
just upgrade it to the real LLM agent with live weather.

---

## Part A — Get the free API keys

### 1. Groq API key (LLM — unlocks the autonomous agent)

1. Go to **https://console.groq.com** and sign in (Google/GitHub works).
2. Open **API Keys** → **Create API Key** → copy it (starts with `gsk_...`).
3. Keep it somewhere safe — you paste it into the Space, never into the code.

Groq's free tier is generous and very fast. If the default model
(`llama-3.3-70b-versatile`) is ever retired, pick a current one from
**https://console.groq.com/docs/models** and set it as `GROQ_MODEL`.

### 2. OpenWeatherMap key (optional — enables *live* weather)

1. Go to **https://home.openweathermap.org/users/sign_up** and create an account.
2. Open **My API keys** and copy the default key.
3. New keys can take a little while (up to ~1–2 hours) to activate. Until then —
   or if you skip this step entirely — the app uses the built-in mock forecast,
   so nothing breaks.

> Without this key the agent still runs end-to-end on realistic mock weather.

---

## Part B — Test locally first (optional but recommended)

```bash
pip install -r requirements.txt

# enable the full agent (optional)
cp .env.example .env
# edit .env and paste your keys

python test_smoke.py        # offline test suite — should print "All smoke tests passed."
python run_evaluation.py    # batch evaluation — writes eval_results.json
python app.py               # opens the Gradio app at http://127.0.0.1:7860
```

---

## Part C — Deploy to Hugging Face Spaces

### Step 1 — Create the Space
1. Sign in at **https://huggingface.co** → click your avatar → **New Space**.
2. Owner = you. Space name = e.g. `travel-weather-agent`.
3. **SDK = Gradio**. Visibility = **Public** (the assignment needs a public link).
4. Click **Create Space**.

### Step 2 — Upload the project files
Upload **all** the project files to the Space (the entire repo):
`app.py`, `agent.py`, `tools.py`, `evaluation.py`, `config.py`,
`run_evaluation.py`, `requirements.txt`, `README.md`, `.gitignore`, and the
`assets/` folder.

Easiest path: on the Space page open **Files** → **Add file** → **Upload files**,
drag everything in, and **Commit**.

> Do **not** upload your `.env` file. Keys go in Settings (Step 3).

Hugging Face needs a YAML header at the top of the Space's `README.md` so it
knows how to run the app. Add this before uploading:

```yaml
---
title: Weather-Aware Travel Planning Agent
emoji: 🌤️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---
```

### Step 3 — Add your keys as secrets
1. In the Space, open **Settings** → **Variables and secrets**.
2. Add a **secret** named `GROQ_API_KEY` with your Groq key.
3. (Optional) add `OPENWEATHER_API_KEY` with your OpenWeatherMap key.
4. (Optional) add `GROQ_MODEL` if you want a non-default model.

### Step 4 — Let it build
The Space installs `requirements.txt` and boots automatically (first build takes
a few minutes). When it shows **Running**, your app is live at:

```
https://huggingface.co/spaces/<your-username>/travel-weather-agent
```

Open it, plan a trip, and confirm the banner reads **FULL AGENT MODE** (if you
added the keys) or **DEMO MODE** (if not).

---

## Part D — Submit (Upload #3)

1. **Blog post:** paste `BLOG_POST.md` into a public Medium post (or keep it in
   the repo as the project documentation). Include the link to the live Space.
2. **Repo / Space link:** the public Hugging Face Space link doubles as both the
   live demo and the code repository (the Files tab shows all source).
3. Add both links to the MS Teams assignment for **Upload #3**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `gradio` import (`HfFolder`) | Already handled — `requirements.txt` pins `huggingface_hub<0.26`. |
| `gr.Dataframe` jinja2 error | Already handled — `requirements.txt` pins `jinja2>=3.1.2`. |
| Banner says DEMO although keys are set | Re-check the secret name is exactly `GROQ_API_KEY`; restart the Space (**Settings → Factory reboot**). |
| Groq model error | The model id changed; set `GROQ_MODEL` to a current one from the Groq models page. |
| Live weather not used | The OpenWeatherMap key may still be activating (up to ~2h) — the app uses mock weather meanwhile. |
