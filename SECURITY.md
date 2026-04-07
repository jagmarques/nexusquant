# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.4.x | Yes |
| < 0.4 | No |

## Reporting a vulnerability

Do not open a public GitHub issue for security vulnerabilities.

Email **security@nexusquant.ai** with:
- A description of the vulnerability
- Steps to reproduce
- Your assessment of impact

You will receive a response within 72 hours. If the issue is confirmed, a patch will be released within 14 days and you will be credited in the release notes unless you prefer otherwise.

This library processes model weights and token data locally. It does not make network requests, store data externally, or execute user-supplied code. The primary attack surface is malformed input tensors passed to the compression pipeline.
