# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately via
[GitHub private vulnerability reporting](https://github.com/lobel-dev/attention-span/security/advisories/new).
Please do not open a public issue for security reports.

## Supported versions

Only the latest release is supported. Installed copies with auto-update
enabled check once a day and move onto it when the check and update succeed.

## Security model

What this tool does, so you can judge reports and behavior against it:

- Reads Claude Code transcript and session files. It never modifies them.
- Writes derived snapshots to an owner-only (`0700`) cache directory under
  `${CLAUDE_HOME:-~/.claude}/hooks/attention-span/cache`.
- Reaches the network in exactly one way: a detached background update pass, at
  most once a day, that issues HTTPS requests to GitHub - the Releases API
  check, and the archive download when a newer release is found. No render ever
  waits on it. `CLAUDE_HEALTH_AUTO_UPDATE=0` disables it.
- Applies an update only after the downloaded archive's contents match the
  sha256 manifest inside it exactly; anything else is rejected and the
  installed version keeps running.

## Install trust

The `curl | bash` installer downloads the `main` branch tarball from GitHub
over TLS - the same trust as cloning the repository and running `install.sh`.
The daily updater then moves installs onto manifest-verified releases.
If you prefer to read before running, clone the repository and run
`bash install.sh` from the checkout.
