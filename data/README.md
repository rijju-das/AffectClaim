# Local data directory

Dataset contents are intentionally excluded from Git. Store only local data or
links created by a documented download/preparation step in this directory.

Recommended local layout:

```text
data/
  raw/<dataset-name>/
  interim/<dataset-name>/
  processed/<dataset-name>/
  manifests/
```

Do not commit images, annotations containing participant identifiers, absolute
machine-specific paths, or derived arrays. Version small schemas and anonymised
example manifests under `examples/` instead.
