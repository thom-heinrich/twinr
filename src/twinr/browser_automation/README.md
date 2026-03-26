# browser_automation

`browser_automation` owns Twinr's versioned browser-automation boundary. The
package intentionally does **not** ship a real browser stack yet. Instead, it
defines the stable request/response contracts and the loader that may attach a
local implementation from the gitignored repo-root `browser_automation/`
workspace.

## Responsibility

`browser_automation` owns:
- define the stable in/out contract for optional browser tasks
- define the driver protocol Twinr code may call
- load one local implementation from the repo-root workspace when explicitly enabled
- fail closed when the feature is disabled, missing, malformed, or outside the configured workspace

`browser_automation` does **not** own:
- Playwright/browser-use/browser-agent implementation details
- voice-policy decisions about which browser tasks Twinr may run
- agent-tool prompting, confirmation flows, or user-facing phrasing

## Local workspace contract

Twinr reserves a gitignored repo-root folder:

```text
browser_automation/
  adapter.py
```

`adapter.py` must export:

```python
def create_browser_automation_driver(*, config, project_root):
    ...
```

The returned object must implement `BrowserAutomationDriver`:

```python
from twinr.browser_automation import BrowserAutomationDriver
```

## Usage

```python
from pathlib import Path

from twinr.agent.base_agent import TwinrConfig
from twinr.browser_automation import BrowserAutomationRequest, load_browser_automation_driver

config = TwinrConfig.from_env(".env")
driver = load_browser_automation_driver(config=config, project_root=Path("."))
result = driver.execute(
    BrowserAutomationRequest(
        task_id="doctor-website-check",
        goal="Check whether the practice site lists holiday opening hours.",
        allowed_domains=("example.org",),
    )
)
```

## See also

- [AGENTS.md](./AGENTS.md)
- [component.yaml](./component.yaml)
- [../README.md](../README.md)
