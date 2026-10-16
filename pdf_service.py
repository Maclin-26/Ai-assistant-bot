import fitz


def extract_text(file_path: str):
    pages = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages