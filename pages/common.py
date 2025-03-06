import hashlib


def _generate_color(token: int):
    """Generate a consistent color for a token"""
    hash_object = hashlib.md5(str(token).encode())
    hash_hex = hash_object.hexdigest()

    r = int(int(hash_hex[0:2], 16) * 0.6 + 256 * 0.4)
    g = int(int(hash_hex[2:4], 16) * 0.6 + 256 * 0.4)
    b = int(int(hash_hex[4:6], 16) * 0.6 + 256 * 0.4)

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    if brightness > 200:
        r = min(r * 0.7, 255)
        g = min(g * 0.7, 255)
        b = min(b * 0.7, 255)

    return f"rgb({r},{g},{b})"


def create_colored_text_html(tokens):
    """Create HTML with color-coded tokens"""
    html_parts = []
    for i, token in enumerate(tokens):
        html_parts.append(
            f'<span style="background-color:{_generate_color(token)}; '
            f"padding: 0; margin-right: 1px; font-size: 1.2em; "
            f'display: inline-block;" '
            f'title="{i}-{token}">{token.replace(" ", "␣")}</span>'
        )
    return "".join(html_parts)
