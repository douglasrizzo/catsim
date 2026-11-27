---
title: Should a broken link fail CI?
kind: decision
status: needs-decision
priority: 3
effort: XS
blocked_by: []
raised_by: [documentation sync (#226)]
---

# Should a broken link fail CI?

## The question

CI's linkcheck step is `continue-on-error: true` (`.github/workflows/ci-cd.yml`). A broken link
shows in the log but never fails a PR. `sphinx/conf.py` already ignores:

- hosts that refuse automated clients (`_REFUSE_AUTOMATED_CLIENTS`);
- targets that only exist once a release deploys (`_PUBLISHED_BY_THE_RELEASE`).

With those ignored, the linkcheck run was clean in #226.

## Options

1. **Fail the build on a broken link.** A dead DOI or a moved page is caught in the PR that would
   publish it. A flaky external host can fail an unrelated PR, so retries or ignores are needed when
   that happens.
2. **Keep `continue-on-error`,** and read the log now and then.
3. **A separate scheduled job** that runs linkcheck weekly and fails on its own, outside PRs.

Recommendation: option 3. Broken links are caught without blocking unrelated PRs.
