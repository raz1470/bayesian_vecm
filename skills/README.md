# Agent skill

`bayesian-vecm/` is an [Agent Skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview): plain markdown that guides an AI agent through a long-term brand analysis with this package. It is written for analysts who have never used a VECM. The agent explains each step, asks at the decisions only the analyst can make and checks the work before anything is reported.

It covers:

- the workflow from the business question to a write-up, with checkpoints;
- the theory in plain English, with a glossary;
- preparing data, including what base sales must be and how to build it with a UCM;
- choosing options, diagnostics, interpretation and pitfalls backed by the package's validation runs;
- a research protocol for judging a funnel metric such as share of search.

## Using it

- **Any agent:** tell it "Read `skills/bayesian-vecm/SKILL.md` in the bayesian_vecm repo and follow it."
- **Claude Code:** copy the folder into your project's `.claude/skills/` (or `~/.claude/skills/` for every project).
- **Claude apps:** zip the `bayesian-vecm` folder and upload it as a skill in Settings.

The skill versions with the package. Use the copy that matches your installed version.
