# NasdaqTrader (Streamlit)

## Run locally

```bash
cd /Users/williamhanks/Developer/projects/test
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m streamlit run test.py --server.address 127.0.0.1 --server.port 8502
```

Then open the URL Streamlit prints (e.g. `http://127.0.0.1:8502`).

## Repository layout

- `test.py` — Streamlit entrypoint
- `online_learner.py` — online learner used by the app
- `clkstrgy/` — optional related scripts (e.g. live chart)

Learner state JSON files are ignored by git by default; they are recreated as you use the app.
