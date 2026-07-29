# Screenshots

Add screenshots of the running app here before publishing to GitHub / LinkedIn. Suggested shots:

1. `chat_ui.png` — the main chat interface with a few messages exchanged (e.g. a greeting, an order status lookup with the ORD12345 flow, and a fallback response).
2. `terminal_cli.png` — the CLI mode (`python src/chatbot.py`) running a short conversation.
3. `training_output.png` — the console output of `python src/train_model.py` showing the classification report.

To capture the chat UI:

```bash
python src/app.py
# open http://localhost:5000 in your browser and take a screenshot
```

Once added, reference them in the main `README.md` under a "Demo" section, e.g.:

```markdown
![Chat UI](screenshots/chat_ui.png)
```
