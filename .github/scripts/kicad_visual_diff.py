#!/usr/bin/env python3
import argparse
import base64
import io
import os
import re
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


WHITE_THRESHOLD = 252
RENDER_LONG_EDGE = 9000

COLOR_BG = (255, 255, 255)
COLOR_UNCHANGED = (205, 205, 205)
COLOR_ADDED = (220, 50, 47)
COLOR_REMOVED = (38, 139, 210)
COLOR_TEXT = (30, 30, 30)
COLOR_MUTED = (90, 90, 90)
COLOR_BORDER = (190, 190, 190)
COLOR_HEADER_BG = (245, 245, 245)

POWER_NAME_RE = re.compile(
    r'^(?:\+\d+(?:\.\d+)?V|\+\d+V\d+|VCC|VDD|VSS|GND|AGND|DGND|PGND|VBAT|VIN|VSYS|3V3|5V|1V8|1V2)$',
    re.IGNORECASE,
)
REF_RE = re.compile(r'^[A-Z]{1,6}\d+[A-Z]?$')


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


FONT_TITLE = load_font(40, bold=True)
FONT_SUB = load_font(28, bold=True)
FONT_BODY = load_font(22, bold=False)
FONT_SMALL = load_font(19, bold=False)
FONT_PAGE = load_font(22, bold=True)


def natural_key(text):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(text))]


def format_list(items, max_items=10):
    items = list(items)
    if not items:
        return "ninguno"
    if len(items) <= max_items:
        return ", ".join(items)
    return f"{', '.join(items[:max_items])} y {len(items) - max_items} más"


def draw_wrapped_text(draw, text, xy, font, fill, max_width, line_spacing=6):
    x, y = xy
    words = text.split()
    lines = []
    current = ""

    for word in words:
        trial = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y


def parse_svg_size(svg_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    root = tree.getroot()

    viewbox = root.attrib.get("viewBox")
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    if viewbox:
        parts = viewbox.replace(",", " ").split()
        if len(parts) == 4:
            _, _, w, h = map(float, parts)
            return w, h

    def parse_dim(v):
        if v is None:
            return None
        cleaned = "".join(ch for ch in v if (ch.isdigit() or ch in ".-"))
        return float(cleaned) if cleaned else None

    w = parse_dim(width)
    h = parse_dim(height)
    if w and h:
        return w, h

    return 4200.0, 2970.0


def compute_render_size(svg_path, long_edge=RENDER_LONG_EDGE):
    w, h = parse_svg_size(svg_path)
    if w >= h:
        out_w = int(long_edge)
        out_h = max(1, int(round(long_edge * h / w)))
    else:
        out_h = int(long_edge)
        out_w = max(1, int(round(long_edge * w / h)))
    return out_w, out_h


def render_svg_to_rgba(svg_path, width, height):
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def fit_image(img: Image.Image, max_w: int, max_h: int):
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS), scale


def wrap_png_into_svg(png_path, svg_path, width, height):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <image href="data:image/png;base64,{b64}" x="0" y="0" width="{width}" height="{height}"/>
</svg>
'''
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


def list_svg_files(directory: str):
    base = Path(directory)
    files = sorted(base.rglob("*.svg"), key=lambda p: natural_key(str(p.relative_to(base))))
    return {str(p.relative_to(base)): p for p in files}


def normalize_page_background(img: Image.Image):
    arr = np.array(img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    flat = rgb.reshape(-1, 3)
    flat_alpha = alpha.reshape(-1)

    quant = ((flat // 4) * 4).astype(np.uint8)
    bright = (flat_alpha > 0) & (quant.mean(axis=1) > 170)

    if not np.any(bright):
        return img

    candidates = quant[bright]
    uniques, counts = np.unique(candidates, axis=0, return_counts=True)
    bg = uniques[np.argmax(counts)].astype(np.int16)

    rgb_i = rgb.astype(np.int16)
    dist = np.max(np.abs(rgb_i - bg[None, None, :]), axis=2)
    rgb[dist <= 12] = 255

    arr[:, :, :3] = rgb
    return Image.fromarray(arr, mode="RGBA")


def mask_from_rgba(img: Image.Image):
    arr = np.array(img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    dist_white = 255 - rgb.mean(axis=2)
    return (alpha > 0) & (dist_white > (255 - WHITE_THRESHOLD))


def max_filter_mask(mask: np.ndarray, passes=1):
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(passes):
        pil = pil.filter(ImageFilter.MaxFilter(3))
    return np.array(pil) > 0


def min_filter_mask(mask: np.ndarray, passes=1):
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(passes):
        pil = pil.filter(ImageFilter.MinFilter(3))
    return np.array(pil) > 0


def cleanup_mask(mask: np.ndarray):
    m = max_filter_mask(mask, 1)
    m = min_filter_mask(m, 1)
    return m


def make_masks(old_img: Image.Image, new_img: Image.Image):
    old_img = normalize_page_background(old_img)
    new_img = normalize_page_background(new_img)

    old_mask = cleanup_mask(mask_from_rgba(old_img))
    new_mask = cleanup_mask(mask_from_rgba(new_img))

    unchanged = old_mask & new_mask
    added_raw = new_mask & (~old_mask)
    removed_raw = old_mask & (~new_mask)

    # engrosar solo nuevo / eliminado
    added = max_filter_mask(added_raw, 2)
    removed = max_filter_mask(removed_raw, 2)

    return unchanged, added, removed


def compose_diff_page(unchanged, added, removed):
    h, w = unchanged.shape
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    arr[unchanged] = np.array(COLOR_UNCHANGED, dtype=np.uint8)
    arr[added] = np.array(COLOR_ADDED, dtype=np.uint8)
    arr[removed] = np.array(COLOR_REMOVED, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def compose_changes_only_page(added, removed):
    h, w = added.shape
    arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    arr[added] = np.array(COLOR_ADDED, dtype=np.uint8)
    arr[removed] = np.array(COLOR_REMOVED, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def crop_to_content(img: Image.Image, threshold=250, pad=120):
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)

    if not np.any(mask):
        return img

    ys, xs = np.where(mask)
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(img.width - 1, xs.max() + pad)
    y2 = min(img.height - 1, ys.max() + pad)

    return img.crop((x1, y1, x2 + 1, y2 + 1))


def compose_page_card(page_img: Image.Image, page_label: str, max_w=3200, max_h=1800):
    fitted, _ = fit_image(page_img, max_w, max_h)

    card_w = fitted.width + 32
    card_h = fitted.height + 64
    card = Image.new("RGB", (card_w, card_h), COLOR_BG)
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=14, fill=COLOR_BG, outline=COLOR_BORDER, width=2)
    draw.rectangle([0, 0, card_w, 44], fill=COLOR_HEADER_BG)
    draw.text((16, 10), page_label, font=FONT_PAGE, fill=COLOR_TEXT)
    card.paste(fitted, (16, 48))

    return card


def stack_cards(cards, gap=26, margin=42):
    width = max(card.width for card in cards) + margin * 2
    height = sum(card.height for card in cards) + gap * (len(cards) - 1) + margin * 2

    canvas = Image.new("RGB", (width, height), COLOR_BG)
    y = margin
    for card in cards:
        x = (width - card.width) // 2
        canvas.paste(card, (x, y))
        y += card.height + gap
    return canvas


def extract_blocks(text, keyword):
    pattern = re.compile(r'\(' + re.escape(keyword) + r'(?=[\s"])')
    pos = 0
    blocks = []

    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        start = m.start()
        depth = 0
        end = None
        for i in range(start, len(text)):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break
        blocks.append(text[start:end])
        pos = end
    return blocks


def match_group(text, pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def parse_at(text):
    m = re.search(r'\(at\s+([-\d\.]+)\s+([-\d\.]+)', text)
    if not m:
        return None
    return (round(float(m.group(1)), 2), round(float(m.group(2)), 2))


def is_power_name(name: str):
    return bool(name and POWER_NAME_RE.match(name.strip()))


def parse_project_semantics(root_sch_path):
    root = Path(root_sch_path)
    project_dir = root.parent

    sch_files = sorted(project_dir.rglob("*.kicad_sch"), key=lambda p: natural_key(str(p.relative_to(project_dir))))

    components = {}
    refs = set()
    power_names = set()
    power_positions = {}
    net_names = set()
    net_positions = {}

    for sch in sch_files:
        rel = str(sch.relative_to(project_dir))
        text = sch.read_text(encoding="utf-8", errors="ignore")

        symbol_blocks = extract_blocks(text, "symbol")
        for block in symbol_blocks:
            uuid = match_group(block, r'\(uuid\s+"([^"]+)"\)')
            lib_id = match_group(block, r'\(lib_id\s+"([^"]+)"\)')
            ref = match_group(block, r'\(property\s+"Reference"\s+"([^"]*)"')
            value = match_group(block, r'\(property\s+"Value"\s+"([^"]*)"')
            at = parse_at(block)

            comp_key = uuid or f"{rel}:{ref}:{lib_id}:{at}"

            components[comp_key] = {
                "sheet": rel,
                "uuid": uuid or "",
                "lib_id": lib_id or "",
                "ref": ref or "",
                "value": value or "",
                "at": at,
            }

            if ref and REF_RE.match(ref) and not ref.startswith("#PWR"):
                refs.add(ref)

            is_power = (lib_id or "").startswith("power:") or (ref or "").startswith("#PWR")
            if is_power:
                pname = (value or "").strip() if value else (lib_id or "").split(":")[-1].strip()
                if pname:
                    power_names.add(pname)
                    power_positions.setdefault(pname, []).append((rel, at))

        for keyword in ("label", "global_label", "hierarchical_label"):
            for block in extract_blocks(text, keyword):
                name = match_group(block, rf'\({keyword}\s+"([^"]+)"')
                at = parse_at(block)
                if not name:
                    continue
                name = name.strip()

                if is_power_name(name):
                    power_names.add(name)
                    power_positions.setdefault(name, []).append((rel, at))
                else:
                    net_names.add(name)
                    net_positions.setdefault(name, []).append((rel, at))

    for k in list(power_positions.keys()):
        power_positions[k] = sorted([x for x in power_positions[k] if x[1] is not None], key=natural_key)
    for k in list(net_positions.keys()):
        net_positions[k] = sorted([x for x in net_positions[k] if x[1] is not None], key=natural_key)

    return {
        "components": components,
        "refs": set(sorted(refs, key=natural_key)),
        "power_names": set(sorted(power_names, key=natural_key)),
        "power_positions": power_positions,
        "net_names": set(sorted(net_names, key=natural_key)),
        "net_positions": net_positions,
    }


def diff_semantics(old_info, new_info):
    old_components = old_info["components"]
    new_components = new_info["components"]

    old_keys = set(old_components.keys())
    new_keys = set(new_components.keys())

    added_components = []
    removed_components = []
    moved_components = []
    changed_values = []
    changed_references = []

    for key in sorted(new_keys - old_keys, key=natural_key):
        comp = new_components[key]
        label = comp["ref"] or comp["value"] or comp["lib_id"] or key
        added_components.append(label)

    for key in sorted(old_keys - new_keys, key=natural_key):
        comp = old_components[key]
        label = comp["ref"] or comp["value"] or comp["lib_id"] or key
        removed_components.append(label)

    for key in sorted(old_keys & new_keys, key=natural_key):
        oldc = old_components[key]
        newc = new_components[key]

        old_label = oldc["ref"] or oldc["value"] or oldc["lib_id"] or key

        if oldc["at"] != newc["at"]:
            moved_components.append(old_label)

        if oldc["value"] != newc["value"]:
            changed_values.append(f"{old_label}: {oldc['value']} -> {newc['value']}")

        if oldc["ref"] != newc["ref"]:
            changed_references.append(f"{oldc['ref']} -> {newc['ref']}")

    added_refs = sorted(new_info["refs"] - old_info["refs"], key=natural_key)
    removed_refs = sorted(old_info["refs"] - new_info["refs"], key=natural_key)

    added_power = sorted(new_info["power_names"] - old_info["power_names"], key=natural_key)
    removed_power = sorted(old_info["power_names"] - new_info["power_names"], key=natural_key)

    added_nets = sorted(new_info["net_names"] - old_info["net_names"], key=natural_key)
    removed_nets = sorted(old_info["net_names"] - new_info["net_names"], key=natural_key)

    modified_nets = []
    modified_power = []

    for name in sorted(old_info["net_names"] & new_info["net_names"], key=natural_key):
        if old_info["net_positions"].get(name, []) != new_info["net_positions"].get(name, []):
            modified_nets.append(name)

    for name in sorted(old_info["power_names"] & new_info["power_names"], key=natural_key):
        if old_info["power_positions"].get(name, []) != new_info["power_positions"].get(name, []):
            modified_power.append(name)

    return {
        "added_components": sorted(added_components, key=natural_key),
        "removed_components": sorted(removed_components, key=natural_key),
        "moved_components": sorted(set(moved_components), key=natural_key),
        "changed_values": sorted(changed_values, key=natural_key),
        "changed_references": sorted(changed_references, key=natural_key),
        "added_refs": added_refs,
        "removed_refs": removed_refs,
        "added_power": added_power,
        "removed_power": removed_power,
        "added_nets": added_nets,
        "removed_nets": removed_nets,
        "modified_nets": modified_nets,
        "modified_power": modified_power,
    }


def build_summary_lines(diff_info):
    lines = []

    if diff_info["added_components"]:
        lines.append(f"componentes agregados: {format_list(diff_info['added_components'])}")
    if diff_info["removed_components"]:
        lines.append(f"componentes eliminados: {format_list(diff_info['removed_components'])}")
    if diff_info["moved_components"]:
        lines.append(f"componentes movidos: {format_list(diff_info['moved_components'])}")

    if diff_info["changed_values"]:
        lines.append(f"valores cambiados: {format_list(diff_info['changed_values'], max_items=6)}")
    if diff_info["changed_references"]:
        lines.append(f"referencias cambiadas: {format_list(diff_info['changed_references'], max_items=6)}")

    if diff_info["added_power"]:
        lines.append(f"power flags agregados: {format_list(diff_info['added_power'])}")
    if diff_info["removed_power"]:
        lines.append(f"power flags eliminados: {format_list(diff_info['removed_power'])}")

    if diff_info["added_nets"]:
        lines.append(f"redes agregadas: {format_list(diff_info['added_nets'])}")
    if diff_info["removed_nets"]:
        lines.append(f"redes eliminadas: {format_list(diff_info['removed_nets'])}")

    if diff_info["modified_nets"]:
        lines.append(f"redes modificadas: {format_list(diff_info['modified_nets'])}")
    if diff_info["modified_power"]:
        lines.append(f"power flags modificados: {format_list(diff_info['modified_power'])}")

    if not lines:
        lines.append("sin cambios semánticos detectables en componentes, referencias, valores, power flags o labels de red")

    return lines


def draw_legend(draw, x, y):
    box = 24
    cx = x
    items = [
        (COLOR_UNCHANGED, "sin cambio"),
        (COLOR_ADDED, "nuevo"),
        (COLOR_REMOVED, "eliminado"),
    ]
    for color, label in items:
        draw.rectangle([cx, y, cx + box, y + box], fill=color, outline=(90, 90, 90), width=1)
        draw.text((cx + box + 10, y - 1), label, font=FONT_SMALL, fill=COLOR_TEXT)
        bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
        cx += box + 10 + (bbox[2] - bbox[0]) + 28


def compose_panel(stacked_img, panel_title, title, author, commit, date, message, summary_lines):
    final_width = 3600
    margin = 42
    header_h = 210
    footer_h = 340

    fitted, _ = fit_image(stacked_img, final_width - 2 * margin, 3200)
    final_height = margin + header_h + fitted.height + 28 + footer_h + margin

    canvas = Image.new("RGB", (final_width, final_height), COLOR_BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, final_width, header_h], fill=COLOR_HEADER_BG)
    draw.text((margin, 28), "KiCad visual diff", font=FONT_TITLE, fill=COLOR_TEXT)
    draw.text((margin, 84), title, font=FONT_SUB, fill=(55, 55, 55))
    draw.text((margin, 126), panel_title, font=FONT_BODY, fill=(75, 75, 75))
    meta = f"autor: {author}   |   fecha: {date}   |   commit: {commit}"
    draw.text((margin, 162), meta, font=FONT_SMALL, fill=COLOR_MUTED)

    y = header_h + margin
    draw_legend(draw, margin, y - 38)

    canvas.paste(fitted, (margin, y))

    fy = y + fitted.height + 24
    draw.line([(margin, fy), (final_width - margin, fy)], fill=COLOR_BORDER, width=1)
    fy += 16

    fy = draw_wrapped_text(draw, f"mensaje: {message}", (margin, fy), FONT_BODY, COLOR_TEXT, final_width - 2 * margin)
    fy += 12

    draw.text((margin, fy), "resumen automático:", font=FONT_BODY, fill=COLOR_TEXT)
    fy += 34

    for line in summary_lines[:10]:
        draw.text((margin + 8, fy), "•", font=FONT_BODY, fill=COLOR_TEXT)
        fy = draw_wrapped_text(draw, line, (margin + 30, fy), FONT_SMALL, COLOR_MUTED, final_width - 2 * margin - 48)
        fy += 4

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-svg-dir", required=True)
    parser.add_argument("--new-svg-dir", required=True)
    parser.add_argument("--old-sch", required=True)
    parser.add_argument("--new-sch", required=True)
    parser.add_argument("--overview-png-out", required=True)
    parser.add_argument("--overview-svg-out", required=True)
    parser.add_argument("--changes-png-out", required=True)
    parser.add_argument("--changes-svg-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--author", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--message", default="")
    parser.add_argument("--title", default="schematic diff")
    args = parser.parse_args()

    old_map = list_svg_files(args.old_svg_dir)
    new_map = list_svg_files(args.new_svg_dir)

    all_keys = sorted(set(old_map.keys()) | set(new_map.keys()), key=natural_key)
    if not all_keys:
        raise SystemExit("No se encontraron SVG para comparar")

    overview_cards = []
    changes_cards = []

    for key in all_keys:
        old_path = old_map.get(key)
        new_path = new_map.get(key)

        ref_path = old_path or new_path
        w, h = compute_render_size(ref_path, long_edge=RENDER_LONG_EDGE)

        if old_path:
            old_img = render_svg_to_rgba(old_path, w, h)
        else:
            old_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))

        if new_path:
            new_img = render_svg_to_rgba(new_path, w, h)
        else:
            new_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))

        unchanged, added, removed = make_masks(old_img, new_img)

        overview_page = compose_diff_page(unchanged, added, removed)
        changes_page = compose_changes_only_page(added, removed)

        overview_cards.append(compose_page_card(overview_page, key))

        has_changes = bool(np.any(added) or np.any(removed))
        if has_changes:
            changes_crop = crop_to_content(changes_page, threshold=250, pad=120)
            changes_cards.append(compose_page_card(changes_crop, key))

    if not changes_cards:
        empty = Image.new("RGB", (1800, 320), COLOR_BG)
        d = ImageDraw.Draw(empty)
        d.rounded_rectangle([0, 0, 1799, 319], radius=12, outline=COLOR_BORDER, width=2, fill=COLOR_BG)
        d.text((40, 70), "No se detectaron cambios visuales suficientemente grandes para recortar.", font=FONT_SUB, fill=COLOR_TEXT)
        d.text((40, 150), "Pero sí puede haber cambios semánticos en componentes, referencias, valores, nets o power flags.", font=FONT_BODY, fill=COLOR_MUTED)
        changes_cards.append(empty)

    overview_stack = stack_cards(overview_cards)
    changes_stack = stack_cards(changes_cards)

    old_info = parse_project_semantics(args.old_sch)
    new_info = parse_project_semantics(args.new_sch)
    diff_info = diff_semantics(old_info, new_info)
    summary_lines = build_summary_lines(diff_info)

    overview_panel = compose_panel(
        stacked_img=overview_stack,
        panel_title="overview contextual",
        title=args.title,
        author=args.author,
        commit=args.commit,
        date=args.date,
        message=args.message,
        summary_lines=summary_lines,
    )

    changes_panel = compose_panel(
        stacked_img=changes_stack,
        panel_title="changes-only · énfasis en nets / power symbols / componentes movidos",
        title=args.title,
        author=args.author,
        commit=args.commit,
        date=args.date,
        message=args.message,
        summary_lines=summary_lines,
    )

    Path(args.overview_png_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.changes_png_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    overview_panel.save(args.overview_png_out, format="PNG", compress_level=1)
    changes_panel.save(args.changes_png_out, format="PNG", compress_level=1)

    wrap_png_into_svg(args.overview_png_out, args.overview_svg_out, overview_panel.width, overview_panel.height)
    wrap_png_into_svg(args.changes_png_out, args.changes_svg_out, changes_panel.width, changes_panel.height)

    with open(args.summary_out, "w", encoding="utf-8") as f:
        f.write("resumen automático de cambios\n")
        f.write("=" * 40 + "\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")

    print(f"overview png: {args.overview_png_out}")
    print(f"overview svg: {args.overview_svg_out}")
    print(f"changes png:  {args.changes_png_out}")
    print(f"changes svg:  {args.changes_svg_out}")
    print(f"summary txt:  {args.summary_out}")


if __name__ == "__main__":
    main()
