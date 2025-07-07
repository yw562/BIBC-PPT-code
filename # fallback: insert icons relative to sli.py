# fallback: insert icons relative to slide top if bullet not found
prs = Presentation('/mnt/data/YUEYI 上会PPT.pptx')
use_slide = None
for slide in prs.slides:
    if any('Use-Case' in (sh.text if sh.has_text_frame else '') for sh in slide.shapes):
        use_slide = slide
        break

# set baseline
top = Inches(2)
left_icon = Inches(0.5)
icons = ["📦", "🍔", "🗑️", "🐕"]
for i, icon in enumerate(icons):
    textbox = use_slide.shapes.add_textbox(left_icon, top + i*Pt(26), Inches(0.4), Inches(0.4))
    tf = textbox.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = icon
    p.font.size = Pt(20)

output_path = '/mnt/data/YUEYI_with_icons_v2.pptx'
prs.save(output_path)
output_path
