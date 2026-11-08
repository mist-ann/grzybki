from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(
    out_path: str | Path,
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any] | None,
    best: dict[str, Any] | None,
    images: dict[str, str | Path],
    title: str | None = None,
) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    instance_name = manifest.get("instance", {}).get("name", "run")
    page_title = title or f"FGO run: {instance_name}"

    image_blocks = []
    for label, image_path in images.items():
        image = Path(image_path)
        try:
            relative = image.relative_to(output.parent)
        except ValueError:
            relative = image
        image_blocks.append(
            f"<figure><figcaption>{html.escape(label)}</figcaption>"
            f'<img src="{html.escape(str(relative))}" alt="{html.escape(label)}">'
            "</figure>"
        )

    manifest_json = html.escape(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    summary_json = html.escape(
        json.dumps(summary or {}, indent=2, ensure_ascii=False, default=str)
    )
    best_json = html.escape(
        json.dumps(best or {}, indent=2, ensure_ascii=False, default=str)
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #222; }}
    h1, h2 {{ line-height: 1.2; }}
    figure {{ margin: 1.5rem 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    figcaption {{ font-weight: 600; margin-bottom: .4rem; }}
    pre {{ background: #f5f5f5; padding: 1rem; overflow: auto; border-radius: 6px; font-size: .85rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(page_title)}</h1>
  <div class="grid">
    {''.join(image_blocks)}
  </div>
  <h2>Summary</h2>
  <pre>{summary_json}</pre>
  <h2>Best solution</h2>
  <pre>{best_json}</pre>
  <h2>Manifest</h2>
  <pre>{manifest_json}</pre>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output
