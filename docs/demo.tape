# mcpeval demo recording
#
# Renders the README demo GIF to docs/demo.gif.
#
# Prerequisites:
#   - vhs:      brew install vhs   (https://github.com/charmbracelet/vhs)
#   - mcpeval:  uv sync            (so `mcpeval` is on PATH, e.g. `uv run`)
#   - API key:  export ANTHROPIC_API_KEY=...   (the eval makes a real call)
#
# Render:
#   vhs docs/demo.tape
#
# Tip: if `mcpeval` isn't on your PATH, replace it below with `uv run mcpeval`.

Output docs/demo.gif

Set Shell zsh
Set FontSize 18
Set Width 1200
Set Height 760
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms

# Clear the prompt without recording the setup
Hide
Type "clear" Enter
Show

Sleep 600ms
Type "mcpeval run evals/examples/incident_response.yaml --threshold 0.85"
Sleep 800ms
Enter

# Hold while the agent runs and the results table renders.
# Bump this up if your eval takes longer than ~8s to complete.
Sleep 9s
