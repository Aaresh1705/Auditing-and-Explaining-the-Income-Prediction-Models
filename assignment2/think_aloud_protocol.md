# Think-Aloud Study Protocol

## Overview

- **Goal:** Surface comprehension failures, confusion, or trust issues with the explanation dashboards.
- **Participants:** 2 fellow students (outside the group).
- **Duration:** ~15-20 minutes per participant.
- **Materials:** One printed/screen-shared stakeholder dashboard (choose one: applicant, director, or data scientist).

---

## Before the Session

1. Choose which dashboard to test (recommend: **Loan Applicant** — most feedback from Assignment 1 role-play was about this one).
2. Prepare a test scenario: a specific applicant profile and their LIME explanation dashboard.
3. Have a note-taking template ready (see below).

---

## Script

### Introduction (~2 min)

> "We're testing how well our explanation dashboard communicates model decisions. We're testing the dashboard, not you — there are no wrong answers. Please think out loud as you look at the dashboard: say whatever comes to mind, what you notice, what confuses you, what you'd want to click on. I'll ask you a few questions along the way."

### Task 1: First Impressions (~3 min)

Show the dashboard. Ask:
> "What is the first thing you notice? What do you think this dashboard is showing you?"

*Observe:* Do they identify the purpose? Do they notice the decision outcome?

### Task 2: Understanding the Decision (~5 min)

> "Based on this dashboard, was this person's loan application approved or rejected? How confident are you in that answer?"

Follow-up:
> "What factors contributed most to this decision?"

*Observe:* Can they read the LIME chart? Do they understand green = positive, red = negative? Do they get confused by any labels?

### Task 3: Actionability (~3 min)

> "If this were your application, what would you do to improve your chances next time?"

*Observe:* Do they find the actionable tips? Do they distinguish controllable vs. non-controllable factors?

### Task 4: Trust (~3 min)

> "How much would you trust this explanation? Is there anything that makes you doubt it?"

Follow-up:
> "Is there anything you'd want to see that isn't shown here?"

### Debrief (~2 min)

> "Any other thoughts? What was the most confusing part? What worked well?"

---

## Observation Template

Copy this for each participant:

```
Participant: [ID / pseudonym]
Date:
Dashboard tested: [which stakeholder / which model variant]
Duration:

### First Impressions
- What did they notice first?
- Did they understand the purpose?

### Decision Comprehension
- Could they identify approved/rejected? [Y/N]
- Could they name contributing factors? [Y/N]
- Specific confusions or misinterpretations:

### Actionability
- Did they find actionable guidance? [Y/N]
- What actions did they mention?
- Confusions:

### Trust
- Trust level expressed: [high/medium/low]
- Reasons for doubt:
- Missing information requested:

### General Observations
- Body language / hesitations:
- Quotes worth noting:
- Suggestions from participant:
```

---

## Analysis Guide

After both sessions, summarize:

1. **Common confusions:** What tripped up both participants?
2. **What worked:** What was immediately understood?
3. **Differences:** Did the two participants react differently? Why?
4. **Design implications:** What specific changes would address the observed confusions?

Link findings explicitly to your "Before vs After" dashboard iteration in the report.
