#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont


def sorted_pages(folder: Path):
    return sorted(folder.glob("page-*.png"))


def open_or_blank(path: Path | None, size=(1600, 1200)):
    if path and path.exists():
        return Image.open(path).convert("RGBA")
    return Image.new("RGBA", size, (255, 255, 255, 255))


def pad_to_same_size(a: Image.Image, b: Image.Image):
    w = max(a.width, b.width)
    h = max(a.height, b.height)

    def pad(img):
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    return pad(a), pad(b)


def make_diff_panel(base: Image.Image, head: Image.Image):
    diff = ImageChops.difference(base.convert("RGB"), head.convert("RGB"))
    gray = diff.convert("L")
    mask = gray.point(lambda p: 255 if p > 10 else 0)

    panel = head.copy()
    overlay = Image.new("RGBA", panel.size, (255, 0, 0, 90))
    overlay.putalpha(mask)
    return Image.alpha_composite(panel, overlay)


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def build_page(base_img, head_img, diff_img, footer_lines, out_path: Path):
    title_h = 42
    footer_h = 120 + 24 * max(0, len(footer_lines) - 3)
    margin = 20
    gap = 20

    panel_w = max(base_img.width, head_img.width, diff_img.width)
    panel_h = max(base_img.height, head_img.height, diff_img.height)

    total_w = margin + panel_w * 3 + gap * 2 + margin
    total_h = margin + title_h + panel_h + gap + footer_h + margin

    canvas = Image.new("RGBA", (total_w, total_h), (248, 248, 248, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(24)
    font_text = load_font(18)

    draw.text((margin, margin), "BASE", fill=(20, 20, 20), font=font_title)
    draw.text((margin + panel_w + gap, margin), "HEAD", fill=(20, 20, 20), font=font_title)
    draw.text((margin + (panel_w + gap) * 2, margin), "DIFF", fill=(20, 20, 20), font=font_title)

    top_y = margin + title_h
    canvas.paste(base_img, (margin, top_y))
    canvas.paste(head_img, (margin + panel_w + gap, top_y))
    canvas.paste(diff_img, (margin + (panel_w + gap) * 2, top_y))

    footer_y = top_y + panel_h + gap
    draw.rounded_rectangle(
        (margin, footer_y, total_w - margin, total_h - margin),
        radius=12,
        fill=(255, 255, 255, 255),
        outline=(210, 210, 210, 255),
        width=1
    )

    y = footer_y + 14
    for line in footer_lines:
        draw.text((margin + 16, y), line, fill=(30, 30, 30), font=font_text)
        y += 24

    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--head-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--message", required=True)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    head_dir = Path(args.head_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pages = sorted_pages(base_dir)
    head_pages = sorted_pages(head_dir)
    count = max(len(base_pages), len(head_pages))

    font_probe = load_font(18)
    probe_img = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe_img)

    for i in range(count):
        base_path = base_pages[i] if i < len(base_pages) else None
        head_path = head_pages[i] if i < len(head_pages) else None

        base_img = open_or_blank(base_path)
        head_img = open_or_blank(head_path, size=(base_img.width, base_img.height))
        base_img, head_img = pad_to_same_size(base_img, head_img)
        diff_img = make_diff_panel(base_img, head_img)

        max_footer_width = base_img.width * 3 + 20 * 2 - 64
        wrapped_message = wrap_text(probe_draw, args.message, font_probe, max_footer_width)

        footer_lines = [
            f"Index: {args.index} | Page: {i+1:02d} | {args.base_sha} -> {args.head_sha}",
            f"Author: {args.author}",
            f"Date: {args.date}",
            "Commit: " + wrapped_message[0],
        ]
        for extra in wrapped_message[1:]:
            footer_lines.append("        " + extra)

        if count == 1:
            out_name = f"{args.index}.png"
        else:
            out_name = f"{args.index}-{i+1:02d}.png"

        build_page(base_img, head_img, diff_img, footer_lines, out_dir / out_name)


if __name__ == "__main__":
    main()
