"""Explore 2025 R2 and R3 PDF structure."""
import pdfplumber

for rnd in [2, 3]:
    pdf_path = f"2025/R{rnd}.pdf"
    print(f"\n{'='*60}")
    print(f"=== 2025/R{rnd}.pdf ===")
    print(f"{'='*60}")
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for page_num in range(min(2, len(pdf.pages))):
            page = pdf.pages[page_num]
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                lines = text.split('\n')
                print(f"\n--- Page {page_num+1} ({len(lines)} lines) ---")
                for i, line in enumerate(lines[:40]):
                    print(f"  L{i:03d}: {repr(line)}")
            else:
                print(f"\n--- Page {page_num+1}: NO TEXT (might be scanned image) ---")
