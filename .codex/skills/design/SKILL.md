---
name: design
description: Design Twinr UI and interaction surfaces for senior-friendly accessibility, clarity, and calm physical-device behavior.
---

# Twinr Design

Use this skill for:

- local web UI changes
- print layout changes
- display-state/eyes/status behavior
- user-facing wording
- physical interaction clarity

This is not a generic startup-app design skill. Twinr is a physical assistant for senior citizens.

## Primary Design Goals

- low cognitive load
- clear state feedback
- high readability
- calm, trustworthy behavior
- minimal steps
- no jargon

## Key Surfaces

- `src/twinr/web/templates/`
  - local dashboard templates
- `src/twinr/web/static/twinr.css`
  - local dashboard styling
- `src/twinr/display/`
  - eyes/status rendering
- `src/twinr/hardware/printer.py`
  - receipt formatting constraints
- `personality/`
  - hidden language/tone context

## Hard Rules

- Do not make the primary flow more complicated.
- Do not introduce app-like navigation complexity into the physical product.
- Keep settings understandable to caregivers and family members.
- Keep printed text short and readable.
- Keep display states easy to interpret at a glance.
- Prefer explicit labels over clever product language.
- User-facing text should sound warm and simple, not technical or “AI-ish”.

## Web UI Guidance

- Prefer server-rendered simplicity over complex frontend behavior.
- Keep forms grouped by real device concerns:
  - connect
  - settings
  - memory
  - personality
  - user
- Each field should map clearly to a real runtime behavior.
- Expose only what an operator can actually understand and use.

## Physical UX Guidance

- Green button means ask/listen. Keep that obvious.
- Yellow button means print. Keep that obvious.
- Status changes should be redundant where possible:
  - sound
  - display
  - spoken answer
  - paper
- Avoid hidden modes that users need to remember.

## Validation

After design-related changes, verify:

1. the primary interaction is still obvious
2. the wording is simple
3. the UI remains readable on desktop and tablet/phone-sized layouts
4. the changed surface still matches the device’s calm and accessible character
