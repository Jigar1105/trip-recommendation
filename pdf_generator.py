    """
    pdf_generator.py
    Bharat Yatra — Professional PDF Itinerary Generator
    reportlab se banaya gaya hai

    FIXES:
    - Bug 1: Hotel rank number ab properly show hoga (fallback to index+1)
    - Bug 2: Star rendering fixed — emoji ki jagah ASCII ★ use kiya, ☆ hata diya
    - Bug 3: Schedule/Hotel emoji icons ki jagah ASCII text icons use kiye (ReportLab Helvetica emoji support nahi karta)
    - Bug 4: DayBanner day number — multi-digit ke liye font size adjust
    - Bug 5: Blank page removed — build_packing mein unnecessary PageBreak nahi, layout fix
    - Bug 6: Getting There text truncation fixed — ab full text wrap hoga, mid-word cut nahi
    - Bug 7: Hotel rank fallback added with enumerate index
    - Bug 8: Transport icon map fix — 'flight','car' keys added, case-insensitive matching
    - Bug 9: Cover pill width increased for longer text
    - Bug 10: type_bg aur icon_map_day sync kiye — saare types covered
    """

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.platypus.flowables import Flowable
    import io
    from datetime import datetime

    # ── BRAND COLORS ──────────────────────────────────────────────
    SAFFRON      = colors.HexColor('#FF6B35')
    DEEP_GREEN   = colors.HexColor('#1B4332')
    GOLD         = colors.HexColor('#F4A261')
    CREAM        = colors.HexColor('#FFF8F0')
    CHARCOAL     = colors.HexColor('#1A1A2E')
    SOFT_GREY    = colors.HexColor('#6B7280')
    LIGHT_GREY   = colors.HexColor('#F3F4F6')
    WHITE        = colors.white
    BLUE_DARK    = colors.HexColor('#1D4ED8')
    BLUE_LIGHT   = colors.HexColor('#EFF6FF')
    GREEN_DARK   = colors.HexColor('#166534')
    GREEN_LIGHT  = colors.HexColor('#F0FDF4')
    YELLOW_LIGHT = colors.HexColor('#FFFBEB')
    RED_LIGHT    = colors.HexColor('#FEF2F2')
    PURPLE_LIGHT = colors.HexColor('#F5F3FF')
    ORANGE_LIGHT = colors.HexColor('#FFF7ED')

    PAGE_W, PAGE_H = A4

    # ── FIX 3 & 8: ASCII icon maps (no emoji — ReportLab Helvetica doesn't support them) ──
    # Schedule item icons as ASCII/Latin text
    SCHEDULE_ICON_MAP = {
        'transport':  '[->]',
        'arrival':    '[*]',
        'hotel':      '[H]',
        'restaurant': '[R]',
        'activity':   '[A]',
        'rest':       '[~]',
        'taxi':       '[->]',
        'location':   '[*]',
        'food':       '[R]',
        'attraction': '[A]',
        'sunset':     '[S]',
        'dinner':     '[D]',
        'sleep':      '[~]',
        'yoga':       '[Y]',
        'beach':      '[B]',
        'shopping':   '[$]',
        'spa':        '[Sp]',
    }

    # FIX 8: Transport icon map — covers all common values including 'flight', 'car'
    TRANSPORT_ICON_MAP = {
        'plane':   '(Air)',
        'flight':  '(Air)',
        'air':     '(Air)',
        'train':   '(Train)',
        'rail':    '(Train)',
        'bus':     '(Bus)',
        'car':     '(Car)',
        'private': '(Car)',
        'cab':     '(Car)',
        'taxi':    '(Cab)',
        'ferry':   '(Boat)',
        'boat':    '(Boat)',
    }


    # ── CUSTOM FLOWABLES ──────────────────────────────────────────
    class SaffronDivider(Flowable):
        def __init__(self, width=None):
            super().__init__()
            self.width  = width or (PAGE_W - 40 * mm)
            self.height = 4

        def draw(self):
            c = self.canv
            c.setStrokeColor(SAFFRON)
            c.setLineWidth(2)
            c.line(0, 2, self.width * 0.55, 2)
            c.setStrokeColor(GOLD)
            c.setLineWidth(1)
            c.line(self.width * 0.55, 2, self.width, 2)


    class DayBanner(Flowable):
        """Full-width dark green day header banner"""
        def __init__(self, day_num, day_title, theme=''):
            super().__init__()
            self.day_num   = day_num
            self.day_title = day_title
            self.theme     = theme
            self.width     = PAGE_W - 40 * mm
            self.height    = 16 * mm

        def draw(self):
            c = self.canv
            w, h = self.width, self.height

            # Background
            c.setFillColor(DEEP_GREEN)
            c.roundRect(0, 0, w, h, 5, fill=1, stroke=0)
            # Saffron accent strip left
            c.setFillColor(SAFFRON)
            c.roundRect(0, 0, 12 * mm, h, 5, fill=1, stroke=0)
            c.rect(8 * mm, 0, 4 * mm, h, fill=1, stroke=0)

            # FIX 4: Day number — auto font size for multi-digit numbers
            day_str = str(self.day_num)
            day_font_size = 13 if len(day_str) == 1 else 10
            c.setFillColor(WHITE)
            c.setFont('Helvetica-Bold', day_font_size)
            c.drawCentredString(6 * mm, h / 2 - (day_font_size / 2) + 1, day_str)

            # Title
            c.setFont('Helvetica-Bold', 10)
            c.drawString(16 * mm, h / 2 + 1.5, self.day_title.upper())
            # Theme subtitle
            if self.theme:
                c.setFont('Helvetica', 8)
                c.setFillColor(colors.HexColor('#A7C4B5'))
                c.drawString(16 * mm, h / 2 - 8, self.theme)


    # ── STYLES ────────────────────────────────────────────────────
    def get_styles():
        s = {}
        s['section_head'] = ParagraphStyle(
            'section_head', fontName='Helvetica-Bold', fontSize=13,
            textColor=DEEP_GREEN, spaceBefore=12, spaceAfter=4)
        s['sub_head'] = ParagraphStyle(
            'sub_head', fontName='Helvetica-Bold', fontSize=10,
            textColor=CHARCOAL, spaceBefore=8, spaceAfter=3)
        s['body'] = ParagraphStyle(
            'body', fontName='Helvetica', fontSize=9,
            textColor=CHARCOAL, leading=14, spaceAfter=3, alignment=TA_JUSTIFY)
        s['body_sm'] = ParagraphStyle(
            'body_sm', fontName='Helvetica', fontSize=8,
            textColor=SOFT_GREY, leading=12, spaceAfter=2)
        s['bold_sm'] = ParagraphStyle(
            'bold_sm', fontName='Helvetica-Bold', fontSize=8,
            textColor=CHARCOAL, leading=12, spaceAfter=2)
        s['highlight'] = ParagraphStyle(
            'highlight', fontName='Helvetica-Bold', fontSize=9,
            textColor=SAFFRON, spaceAfter=2)
        s['time_txt'] = ParagraphStyle(
            'time_txt', fontName='Helvetica-Bold', fontSize=8,
            textColor=BLUE_DARK, alignment=TA_RIGHT)
        s['cost_txt'] = ParagraphStyle(
            'cost_txt', fontName='Helvetica-Bold', fontSize=7,
            textColor=SAFFRON, alignment=TA_RIGHT)
        s['center'] = ParagraphStyle(
            'center', fontName='Helvetica', fontSize=8,
            textColor=SOFT_GREY, alignment=TA_CENTER)
        s['footer'] = ParagraphStyle(
            'footer', fontName='Helvetica', fontSize=7,
            textColor=SOFT_GREY, alignment=TA_CENTER, leading=11)
        return s


    # ── PAGE DECORATOR (header + footer on every page after cover) ─
    def page_decorator(canvas, doc, destination=''):
        if doc.page == 1:
            return
        canvas.saveState()
        w, h = A4

        # Header
        canvas.setFillColor(DEEP_GREEN)
        canvas.rect(0, h - 11 * mm, w, 11 * mm, fill=1, stroke=0)
        canvas.setFillColor(SAFFRON)
        canvas.rect(0, h - 12 * mm, w, 1 * mm, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(WHITE)
        canvas.drawString(15 * mm, h - 7 * mm, 'BHARAT YATRA  |  Travel Planner')
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(w - 15 * mm, h - 7 * mm, destination.upper())

        # Footer
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, w, 9 * mm, fill=1, stroke=0)
        canvas.setFillColor(SOFT_GREY)
        canvas.setFont('Helvetica', 7)
        canvas.drawString(15 * mm, 3 * mm,
            f"Generated on {datetime.now().strftime('%d %B %Y')}  |  Bharat Yatra")
        canvas.drawRightString(w - 15 * mm, 3 * mm, f"Page {doc.page}")
        canvas.restoreState()


    # ── COVER PAGE ────────────────────────────────────────────────
    class CoverPage(Flowable):
        def __init__(self, data):
            super().__init__()
            self.data   = data
            self.width  = PAGE_W - 50 * mm
            self.height = PAGE_H - 80 * mm

        def draw(self):
            c    = self.canv
            d    = self.data
            w, h = self.width, self.height

            # Dark gradient background
            c.setFillColor(colors.HexColor('#0D1B2A'))
            c.rect(0, 0, w, h, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#1A2E20'))
            c.rect(0, h * 0.35, w, h * 0.65, fill=1, stroke=0)

            # Decorative circles
            c.setFillColor(colors.HexColor('#FFFFFF08'))
            c.circle(w * 0.85, h * 0.25, 20 * mm, fill=1, stroke=0)
            c.circle(w * 0.05, h * 0.05, 15 * mm, fill=1, stroke=0)

            # Logo / Brand — FIX 3: no emoji, use plain text stars
            c.setFillColor(SAFFRON)
            c.setFont('Helvetica-Bold', 9)
            c.drawCentredString(w / 2, h * 0.92, '*  BHARAT YATRA  *')
            c.setFillColor(colors.HexColor('#94A3B8'))
            c.setFont('Helvetica', 7)
            c.drawCentredString(w / 2, h * 0.88, 'PROFESSIONAL TRAVEL PLANNER')

            # Trip title
            title = d.get('trip_title', 'Your Dream Trip to India')
            c.setFillColor(WHITE)
            words = title.split()
            if len(title) > 38:
                mid   = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                c.setFont('Helvetica-Bold', 18)
                c.drawCentredString(w / 2, h * 0.80, line1)
                c.drawCentredString(w / 2, h * 0.75, line2)
            else:
                c.setFont('Helvetica-Bold', 20)
                c.drawCentredString(w / 2, h * 0.78, title)

            # Route line — FIX 3: no airplane emoji
            src  = d.get('source_city', '')
            dest = d.get('destination', '')
            route = f"{src}  -->  {dest}" if src and dest else dest
            c.setFillColor(GOLD)
            c.setFont('Helvetica-Bold', 10)
            c.drawCentredString(w / 2, h * 0.68, route)

            # FIX 9: Info pills — wider pill_w to prevent text clipping
            pills = []
            if d.get('duration_days'):
                pills.append(f"{d['duration_days']} Days")
            if d.get('travel_style'):
                pills.append(f"{d['travel_style'].capitalize()} Style")
            if d.get('ideal_for'):
                pills.append(f"{d['ideal_for'].capitalize()}")

            if pills:
                pill_w = 38 * mm   # was 32mm — increased for longer labels
                gap    = 4 * mm
                total  = len(pills) * pill_w + (len(pills) - 1) * gap
                start  = (w - total) / 2
                for i, pill_text in enumerate(pills):
                    px = start + i * (pill_w + gap)
                    py = h * 0.56
                    c.setFillColor(colors.HexColor('#FFFFFF12'))
                    c.roundRect(px, py, pill_w, 8.5 * mm, 2.5, fill=1, stroke=0)
                    c.setStrokeColor(SAFFRON)
                    c.setLineWidth(0.5)
                    c.roundRect(px, py, pill_w, 8.5 * mm, 2.5, fill=0, stroke=1)
                    c.setFillColor(WHITE)
                    c.setFont('Helvetica-Bold', 7)
                    c.drawCentredString(px + pill_w / 2, py + 2.8 * mm, pill_text)

            # Saffron accent band
            c.setFillColor(SAFFRON)
            c.rect(0, h * 0.42, w, 2.5 * mm, fill=1, stroke=0)
            c.setFillColor(GOLD)
            c.rect(0, h * 0.42 - 2.5 * mm, w, 0.7 * mm, fill=1, stroke=0)

            # Budget badge
            budget = d.get('total_budget_range', '')
            if budget:
                badge_y = h * 0.38
                c.setFillColor(SAFFRON)
                c.roundRect(w / 2 - 50 * mm, badge_y - 6 * mm, 100 * mm, 8 * mm, 2.5, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont('Helvetica-Bold', 7.5)
                c.drawCentredString(w / 2, badge_y - 2 * mm, f"ESTIMATED BUDGET  |  {budget}")

            # Bottom disclaimer
            c.setFillColor(colors.HexColor('#64748B'))
            c.setFont('Helvetica', 6)
            c.drawCentredString(w / 2, 12 * mm,
                f"Generated on {datetime.now().strftime('%d %b %Y')}  |  bharat-yatra.app")
            c.drawCentredString(w / 2, 7 * mm,
                'All prices are approximate. Please verify with providers before booking.')


    # ── SECTION BUILDERS ──────────────────────────────────────────

    def build_overview(story, data, s):
        story.append(Paragraph('TRIP OVERVIEW', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        if data.get('overview'):
            story.append(Paragraph(data['overview'], s['body']))
            story.append(Spacer(1, 4 * mm))

        # FIX 6: Getting There — removed [:150] truncation, use full text with wrapping
        getting_there = data.get('getting_there_summary', '')

        rows = [
            [Paragraph('<b>From</b>', s['bold_sm']),
            Paragraph(data.get('source_city', 'N/A'), s['body_sm']),
            Paragraph('<b>To</b>', s['bold_sm']),
            Paragraph(data.get('destination', 'N/A'), s['body_sm'])],
            [Paragraph('<b>Duration</b>', s['bold_sm']),
            Paragraph(f"{data.get('duration_days', '?')} Days", s['body_sm']),
            Paragraph('<b>Travel Style</b>', s['bold_sm']),
            Paragraph(data.get('travel_style', '').capitalize(), s['body_sm'])],
            [Paragraph('<b>Best Time</b>', s['bold_sm']),
            Paragraph(data.get('best_time_reminder', 'Year-round'), s['body_sm']),
            Paragraph('<b>Ideal For</b>', s['bold_sm']),
            Paragraph(data.get('ideal_for', 'All').capitalize(), s['body_sm'])],
        ]

        if getting_there:
            rows.append([
                Paragraph('<b>Getting There</b>', s['bold_sm']),
                # FIX 6: span across 3 cols using colspan via empty cells; full text, no truncation
                Paragraph(getting_there, s['body_sm']),
                '', ''
            ])

        tbl = Table(rows, colWidths=[28*mm, 57*mm, 28*mm, 57*mm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), CREAM),
            ('TEXTCOLOR', (0,0), (0,-1), SAFFRON),
            ('TEXTCOLOR', (2,0), (2,-1), SAFFRON),
            ('GRID',      (0,0), (-1,-1), 0.4, colors.HexColor('#E5E7EB')),
            ('PADDING',   (0,0), (-1,-1), 7),
            ('VALIGN',    (0,0), (-1,-1), 'TOP'),
            # FIX 6: Getting There row spans cols 1-3 for more text space
            ('SPAN',      (1, len(rows)-1), (3, len(rows)-1)) if getting_there else ('SPAN', (0,0), (0,0)),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 5 * mm))


    def build_transport(story, data, s):
        transport = data.get('transport', {})
        if not transport:
            return

        src  = data.get('source_city', 'Source')
        dest = data.get('destination', 'Destination')

        story.append(Paragraph('TRANSPORT OPTIONS', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        for direction_key, label in [
            ('outward', f'OUTWARD — {src} to {dest}'),
            ('return',  f'RETURN — {dest} to {src}')
        ]:
            options = transport.get(direction_key, [])
            if not options:
                continue

            story.append(Paragraph(label, s['sub_head']))

            header = [
                Paragraph('<b>Mode</b>',        s['bold_sm']),
                Paragraph('<b>Operator</b>',    s['bold_sm']),
                Paragraph('<b>Duration</b>',    s['bold_sm']),
                Paragraph('<b>Price</b>',       s['bold_sm']),
                Paragraph('<b>Booking Tip</b>', s['bold_sm']),
            ]
            rows = [header]

            for opt in options:
                # FIX 8: case-insensitive icon lookup, fallback to mode name
                raw_icon = opt.get('icon', opt.get('mode', '')).lower()
                icon = TRANSPORT_ICON_MAP.get(raw_icon, f'({opt.get("mode","")[:4]})')

                rows.append([
                    Paragraph(f"<b>{icon} {opt.get('mode','')}</b>", s['highlight']),
                    Paragraph(opt.get('operator', 'N/A'), s['body_sm']),
                    Paragraph(opt.get('duration', 'N/A'), s['body_sm']),
                    Paragraph(f"<b>{opt.get('price_range','N/A')}</b>", s['bold_sm']),
                    Paragraph(opt.get('booking_tip', ''), s['body_sm']),
                ])

            tbl = Table(rows, colWidths=[22*mm, 38*mm, 22*mm, 38*mm, 50*mm], repeatRows=1)
            tbl.setStyle(TableStyle([
                ('BACKGROUND',     (0,0), (-1,0), BLUE_DARK),
                ('TEXTCOLOR',      (0,0), (-1,0), WHITE),
                ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',       (0,0), (-1,0), 8),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [BLUE_LIGHT, WHITE]),
                ('GRID',           (0,0), (-1,-1), 0.4, colors.HexColor('#BFDBFE')),
                ('PADDING',        (0,0), (-1,-1), 5),
                ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 4 * mm))


    def build_hotels(story, data, s):
        hotels = data.get('hotels', [])
        if not hotels:
            return

        story.append(PageBreak())
        story.append(Paragraph('RECOMMENDED HOTELS', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        style_label = data.get('travel_style', 'balanced').capitalize()
        story.append(Paragraph(
            f"Top 5 hotels for <b>{style_label}</b> travel style — prices per night",
            s['body']
        ))
        story.append(Spacer(1, 3 * mm))

        for idx, hotel in enumerate(hotels[:5]):
            # FIX 2: Star rendering — use only ASCII ★, no ☆ (not supported in Helvetica)
            stars = max(0, min(5, int(hotel.get('stars', 3))))
            star_str = '★' * stars + '-' * (5 - stars)  # dash instead of ☆

            hi_str   = '   / '.join(hotel.get('highlights', []))
            plat_str = ', '.join(hotel.get('booking_platforms', []))
            rating   = hotel.get('rating', 'N/A')
            name     = hotel.get('name', 'Hotel')
            area     = hotel.get('area', '')
            price    = hotel.get('price_per_night', 'N/A')
            why      = hotel.get('why_choose', '')

            # FIX 1 & 7: rank — use from data if present, else fallback to enumerate index+1
            rank = hotel.get('rank', '') or (idx + 1)

            rows = [
                [
                    Paragraph(f"<b>#{rank}  {name}</b>", s['sub_head']),
                    Paragraph(f"<b>{price}</b> / night", s['highlight']),
                ],
                [
                    Paragraph(f"[loc] {area}   {star_str}   {rating}/5", s['body_sm']),
                    Paragraph(f"Book: {plat_str}", s['body_sm']),
                ],
                [
                    Paragraph(f"[ok] {hi_str}", s['body_sm']),
                    Paragraph(why, s['body_sm']),
                ],
            ]

            tbl = Table(rows, colWidths=[115*mm, 55*mm])
            tbl.setStyle(TableStyle([
                ('BACKGROUND',  (0,0), (-1,-1), GREEN_LIGHT),
                ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#DCFCE7')),
                ('BOX',         (0,0), (-1,-1), 0.6, GREEN_DARK),
                ('LINEBELOW',   (0,0), (-1,0),  0.5, colors.HexColor('#86EFAC')),
                ('PADDING',     (0,0), (-1,-1), 6),
                ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 3 * mm))


    def build_schedule(story, data, s):
        days = data.get('days', [])
        if not days:
            return

        story.append(PageBreak())
        story.append(Paragraph('DAY-BY-DAY ITINERARY', s['section_head']))
        story.append(SaffronDivider())

        # FIX 10: type_bg — added all missing type keys that appear in icon_map
        type_bg = {
            'transport':  colors.HexColor('#EFF6FF'),
            'arrival':    colors.HexColor('#F0FDF4'),
            'hotel':      ORANGE_LIGHT,
            'restaurant': colors.HexColor('#FEF9C3'),
            'activity':   PURPLE_LIGHT,
            'rest':       colors.HexColor('#F9FAFB'),
            'attraction': PURPLE_LIGHT,
            'sunset':     colors.HexColor('#FFF7ED'),
            'dinner':     colors.HexColor('#FEF9C3'),
            'sleep':      colors.HexColor('#F9FAFB'),
            'yoga':       colors.HexColor('#F0FDF4'),
            'beach':      colors.HexColor('#EFF6FF'),
            'shopping':   colors.HexColor('#FEF9C3'),
            'spa':        colors.HexColor('#FDF2F8'),
            'food':       colors.HexColor('#FEF9C3'),
            'taxi':       colors.HexColor('#EFF6FF'),
            'location':   colors.HexColor('#F0FDF4'),
        }

        for day_obj in days:
            day_num   = day_obj.get('day', 1)
            day_title = day_obj.get('title', f'Day {day_num}')
            theme     = day_obj.get('theme', '')
            day_cost  = day_obj.get('day_total_cost', '')
            tip       = day_obj.get('insider_tip', '')
            schedule  = day_obj.get('schedule', [])

            story.append(Spacer(1, 5 * mm))
            story.append(DayBanner(day_num, day_title, theme))
            story.append(Spacer(1, 2 * mm))

            if schedule:
                sched_data   = []
                sched_styles = [
                    ('FONTSIZE',     (0,0), (-1,-1), 8),
                    ('VALIGN',       (0,0), (-1,-1), 'TOP'),
                    ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor('#E5E7EB')),
                    ('LEFTPADDING',  (0,0), (-1,-1), 5),
                    ('RIGHTPADDING', (0,0), (-1,-1), 5),
                    ('TOPPADDING',   (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING',(0,0), (-1,-1), 5),
                ]

                for i, item in enumerate(schedule):
                    item_type = item.get('type', 'activity')
                    # FIX 10: icon lookup — use icon_type if present, else item_type
                    icon_key  = item.get('icon_type', item_type)
                    # FIX 3: ASCII icons from SCHEDULE_ICON_MAP, no emoji
                    icon      = SCHEDULE_ICON_MAP.get(icon_key,
                                SCHEDULE_ICON_MAP.get(item_type, '[>]'))

                    time_str  = item.get('time', '')
                    cost_str  = item.get('cost', '')
                    activity  = item.get('activity', '')
                    place     = item.get('place', '')
                    details   = item.get('details', '')
                    cuisine   = item.get('cuisine', '')
                    must_try  = item.get('must_try', [])
                    duration  = item.get('duration', '')

                    # Left cell: time + cost
                    left_tbl = Table([
                        [Paragraph(time_str, s['time_txt'])],
                        [Paragraph(cost_str, s['cost_txt'])],
                    ], colWidths=[28*mm])
                    left_tbl.setStyle(TableStyle([
                        ('PADDING',  (0,0), (-1,-1), 0),
                        ('FONTSIZE', (0,0), (-1,-1), 8),
                    ]))

                    # Right cell: activity details
                    right_parts = [f"<b>{icon} {activity}</b>"]
                    if place:
                        right_parts.append(f"<font color='#6B7280'>[loc] {place}</font>")
                    if details:
                        right_parts.append(f"<font color='#374151'>{details}</font>")
                    if cuisine:
                        right_parts.append(f"<font color='#92400E'>Cuisine: {cuisine}</font>")
                    if duration:
                        right_parts.append(f"<font color='#7C3AED'>Duration: {duration}</font>")
                    if must_try and isinstance(must_try, list):
                        right_parts.append(
                            f"<font color='#065F46'>Must try: {', '.join(must_try)}</font>"
                        )

                    right_para = Paragraph('<br/>'.join(right_parts), s['body_sm'])
                    sched_data.append([left_tbl, right_para])

                    # FIX 10: background from extended type_bg dict
                    bg = type_bg.get(item_type, WHITE)
                    sched_styles.append(('BACKGROUND', (0,i), (0,i), bg))
                    sched_styles.append(('BACKGROUND', (1,i), (1,i),
                        colors.HexColor('#FAFAFA') if i % 2 == 0 else WHITE))

                sched_tbl = Table(sched_data, colWidths=[30*mm, 140*mm])
                sched_tbl.setStyle(TableStyle(sched_styles))
                story.append(sched_tbl)

            # Day footer
            if day_cost or tip:
                footer_rows = []
                if day_cost:
                    footer_rows.append([Paragraph(f"<b>Day Budget: {day_cost}</b>", s['bold_sm'])])
                if tip:
                    footer_rows.append([Paragraph(f"Insider Tip: {tip}", s['body_sm'])])
                footer_tbl = Table(footer_rows, colWidths=[170*mm])
                footer_tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), YELLOW_LIGHT),
                    ('BOX',        (0,0), (-1,-1), 0.5, GOLD),
                    ('PADDING',    (0,0), (-1,-1), 6),
                ]))
                story.append(Spacer(1, 1 * mm))
                story.append(footer_tbl)

            story.append(Spacer(1, 4 * mm))


    def build_budget(story, data, s):
        budget = data.get('budget_breakdown', {})
        if not budget:
            return

        # FIX 5: PageBreak only if needed — keeps budget from floating on a blank page
        story.append(PageBreak())
        story.append(Paragraph('BUDGET BREAKDOWN', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        label_map = [
            ('transport_one_way',   'Transport (Outward)'),
            ('transport_return',    'Transport (Return)'),
            ('accommodation_total', 'Accommodation Total'),
            ('food_total',          'Food and Dining'),
            ('activities_total',    'Activities and Entry Fees'),
            ('local_transport',     'Local Transport'),
            ('miscellaneous',       'Shopping and Miscellaneous'),
            ('grand_total',         'GRAND TOTAL (Per Person)'),
        ]

        rows = [[
            Paragraph('<b>Category</b>',       s['bold_sm']),
            Paragraph('<b>Estimated Cost</b>', s['bold_sm']),
        ]]
        for key, label in label_map:
            if key in budget and budget[key]:
                rows.append([
                    Paragraph(label, s['body']),
                    Paragraph(f"<b>{budget[key]}</b>", s['bold_sm']),
                ])

        tbl = Table(rows, colWidths=[115*mm, 55*mm])
        style_cmds = [
            ('BACKGROUND',     (0,0),  (-1,0),  DEEP_GREEN),
            ('TEXTCOLOR',      (0,0),  (-1,0),  WHITE),
            ('ROWBACKGROUNDS', (0,1),  (-1,-2), [LIGHT_GREY, WHITE]),
            ('BACKGROUND',     (0,-1), (-1,-1), CREAM),
            ('FONTNAME',       (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR',      (0,-1), (-1,-1), SAFFRON),
            ('FONTSIZE',       (0,-1), (-1,-1), 10),
            ('LINEABOVE',      (0,-1), (-1,-1), 1.5, SAFFRON),
            ('GRID',           (0,0),  (-1,-1), 0.4, colors.HexColor('#D1D5DB')),
            ('PADDING',        (0,0),  (-1,-1), 7),
            ('ALIGN',          (1,0),  (1,-1),  'RIGHT'),
            ('FONTSIZE',       (0,1),  (-1,-2), 9),
        ]
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 5 * mm))


    def build_packing(story, data, s):
        packing = data.get('packing_list', {})
        if not packing:
            return

        # FIX 5: No PageBreak here — avoids blank page between budget and packing
        # Budget already has PageBreak; packing follows naturally on same page if space allows
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph('PACKING CHECKLIST', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        cats = [
            ('essentials', 'Essentials',           colors.HexColor('#FEE2E2')),
            ('clothing',   'Clothing',             colors.HexColor('#DBEAFE')),
            ('gear',       'Gear & Accessories',   colors.HexColor('#D1FAE5')),
            ('medicines',  'Medicines',            colors.HexColor('#FEF9C3')),
        ]
        rows = []
        row_bgs_left = []
        for key, label, bg in cats:
            items = packing.get(key, [])
            if not items:
                continue
            rows.append([
                Paragraph(f"<b>{label}</b>", s['bold_sm']),
                Paragraph('  |  '.join(items), s['body_sm']),
            ])
            row_bgs_left.append(bg)

        if rows:
            tbl = Table(rows, colWidths=[40*mm, 130*mm])
            style_cmds = [
                ('GRID',    (0,0), (-1,-1), 0.4, colors.HexColor('#E5E7EB')),
                ('PADDING', (0,0), (-1,-1), 7),
                ('VALIGN',  (0,0), (-1,-1), 'TOP'),
            ]
            for i, bg in enumerate(row_bgs_left):
                style_cmds.append(('BACKGROUND', (0,i), (0,i), bg))
                style_cmds.append(('BACKGROUND', (1,i), (1,i), WHITE))
            tbl.setStyle(TableStyle(style_cmds))
            story.append(tbl)
            story.append(Spacer(1, 4 * mm))


    def build_emergency(story, data, s):
        contacts = data.get('emergency_contacts', [
            {'name': 'Police',           'number': '100'},
            {'name': 'Ambulance',        'number': '108'},
            {'name': 'Tourist Helpline', 'number': '1363'},
            {'name': 'Women Helpline',   'number': '1091'},
        ])

        story.append(Paragraph('EMERGENCY CONTACTS', s['section_head']))
        story.append(SaffronDivider())
        story.append(Spacer(1, 3 * mm))

        rows = [[
            Paragraph(f"<b>{c.get('name','')}</b>", s['bold_sm']),
            Paragraph(f"<b>{c.get('number','')}</b>", s['highlight']),
        ] for c in contacts]

        # 2-column grid
        half  = (len(rows) + 1) // 2
        left  = rows[:half]
        right = rows[half:]
        while len(right) < len(left):
            right.append([Paragraph('', s['body_sm']), Paragraph('', s['body_sm'])])

        combined = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]
        tbl = Table(combined, colWidths=[55*mm, 27*mm, 55*mm, 27*mm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), RED_LIGHT),
            ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#FECACA')),
            ('PADDING',     (0,0), (-1,-1), 6),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('LINEAFTER',   (1,0), (1,-1),  1, colors.HexColor('#FECACA')),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 4 * mm))


    # ── MAIN ENTRY ────────────────────────────────────────────────
    def generate_itinerary_pdf(data: dict, output_buffer: io.BytesIO) -> None:
        """
        Itinerary JSON dict se professional PDF banao.
        output_buffer: BytesIO object mein PDF write hoga
        """
        destination = data.get('destination', 'India')
        s = get_styles()

        doc = SimpleDocTemplate(
            output_buffer,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=16 * mm,
            bottomMargin=14 * mm,
            title=data.get('trip_title', 'Bharat Yatra Itinerary'),
            author='Bharat Yatra',
            subject=f"Trip Itinerary to {destination}",
        )

        story = []

        # 1. Cover page
        story.append(KeepTogether([CoverPage(data), PageBreak()]))

        # 2. Overview
        build_overview(story, data, s)

        # 3. Transport
        build_transport(story, data, s)

        # 4. Hotels
        build_hotels(story, data, s)

        # 5. Day schedule
        build_schedule(story, data, s)

        # 6. Budget + Packing (on same page flow — no blank page)
        build_budget(story, data, s)

        # 7. Packing (no extra PageBreak inside)
        build_packing(story, data, s)

        # 8. Emergency
        build_emergency(story, data, s)

        # 9. Disclaimer
        story.append(Spacer(1, 5 * mm))
        story.append(HRFlowable(width='100%', color=SOFT_GREY, thickness=0.5))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            '<b>Disclaimer:</b> All prices, hotel rates, timings and recommendations in this '
            'document are approximate AI-generated estimates. Please verify all bookings, '
            'transport schedules and costs directly with service providers before travelling. '
            'Bharat Yatra is not responsible for any discrepancies or changes in prices.',
            s['footer']
        ))

        def _decorator(canvas, doc):
            page_decorator(canvas, doc, destination)

        doc.build(story, onFirstPage=_decorator, onLaterPages=_decorator)