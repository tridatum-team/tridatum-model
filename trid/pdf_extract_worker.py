"""Isolated, bounded PDF text extraction worker.

Run as `python -I -m trid.pdf_extract_worker MAX_BYTES MAX_PAGES MAX_CHARACTERS CPU_SECONDS ADDRESS_SPACE_BYTES`.
Reads the PDF from stdin and prints one JSON object: {'pages', 'warnings'} or {'error'}.
"""
import json
import logging
import sys


def _size_label(max_bytes):
    mib = max_bytes / (1024 * 1024)
    return f'{int(mib)}MB' if mib == int(mib) else f'{mib:.1f}MB'


def _chars_label(max_characters):
    if max_characters % 10000 == 0:
        return f'{max_characters // 10000}만 자'
    return f'{max_characters:,}자'


def main(argv=None):
    max_bytes, max_pages, max_characters, cpu_seconds, address_space = map(int, (sys.argv[1:] if argv is None else argv))
    if sys.platform != 'win32':
        import resource
        # CPU and address-space limits supplement the parent's wall-clock timeout.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if sys.platform != 'darwin':
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    logging.disable(logging.CRITICAL)
    from trid import pdf_documents as pdf
    pdf.MAX_BYTES = max_bytes
    pdf.MAX_PAGES = max_pages
    pdf.MAX_CHARACTERS = max_characters
    try:
        raw = sys.stdin.buffer.read(max_bytes + 1)
        pages, warnings = pdf.extract_pdf(raw, 'document.pdf')
        print(json.dumps({'pages': pages, 'warnings': warnings}))
    except Exception:
        message = f'텍스트가 있는 PDF {_size_label(max_bytes)}·{max_pages}페이지·{_chars_label(max_characters)} 이하로 다시 시도해 주세요.'
        print(json.dumps({'error': message}))


if __name__ == '__main__':
    main()
