from pathlib import Path
import re
import shutil

path = Path("/workspace/telerobot/src/telerobot/web_ui/index.html")
backup = Path("/workspace/telerobot/src/telerobot/web_ui/index.html.before_rif_text")

shutil.copy2(path, backup)

html = path.read_text()

# Remove any existing RIF banner/entity we previously added
html = re.sub(
    r'\s*<!-- =+ -->\s*'
    r'<!-- RIF ROBOTICS HEADER / BANNER -->.*?'
    r'<!-- END RIF ROBOTICS BANNER\s*-->\s*'
    r'<!-- =+ -->',
    '',
    html,
    flags=re.DOTALL
)

# Also remove older simple rif-banner entity if present
html = re.sub(
    r'\s*<!-- RIF Robotics Banner -->\s*'
    r'<a-entity[^>]*id="rif-banner".*?</a-entity>',
    '',
    html,
    flags=re.DOTALL
)

rif_text = '''
      <!-- RIF Robotics -->
      <a-text
        id="rif-banner"
        value="RIF ROBOTICS"
        position="0 2.25 -3"
        align="center"
        anchor="center"
        color="#FFFFFF"
        width="4"
        look-at-headset>
      </a-text>
'''

marker = "<a-assets></a-assets>"

if 'id="rif-banner"' not in html:
    html = html.replace(marker, marker + "\n" + rif_text, 1)

path.write_text(html)

print("✅ Added RIF ROBOTICS text only")
print(f"✅ Updated: {path}")
print(f"✅ Backup:  {backup}")
