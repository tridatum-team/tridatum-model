"""Parse untrusted PDFs in a separate, resource-limited process."""
from dataclasses import dataclass
import json
import subprocess
import sys
import threading
import time

from trid.pdf_documents import Document, DocumentError, make_chunks


@dataclass(frozen=True)
class PdfLimits:
    max_bytes: int = 2 * 1024 * 1024
    max_pages: int = 20
    max_characters: int = 50_000
    wall_seconds: float = 15.0
    cpu_seconds: int = 10
    address_space_bytes: int = 1024 ** 3


DEMO_LIMITS = PdfLimits()


def parse_pdf(body, name, tok, limits=DEMO_LIMITS):
    # Parsing untrusted PDFs happens in a separate process with a wall-clock timeout.
    try:
        process = subprocess.Popen(
            [sys.executable, '-I', '-m', 'trid.pdf_extract_worker', str(limits.max_bytes), str(limits.max_pages),
             str(limits.max_characters), str(limits.cpu_seconds), str(limits.address_space_bytes)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        output = []
        def communicate():
            output.append(process.communicate(body)[0])
        communication = threading.Thread(target=communicate, daemon=True)
        communication.start()
        deadline = time.monotonic() + limits.wall_seconds
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise DocumentError('PDF 처리 시간이 초과됐습니다. 더 작은 문서를 넣어 주세요.')
                if sys.platform == 'darwin':
                    memory = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, timeout=2).stdout.strip()
                    if memory and int(memory) > 512 * 1024:
                        raise DocumentError('PDF 처리 메모리 한도를 넘었습니다. 더 작은 문서를 넣어 주세요.')
                communication.join(.1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            communication.join(3)
        if process.returncode or not output:
            raise DocumentError('PDF를 읽지 못했습니다. 작은 텍스트 PDF로 다시 시도해 주세요.')
        result = json.loads(output[0])
    except DocumentError:
        raise
    except (subprocess.SubprocessError, ValueError):
        raise DocumentError('PDF를 읽지 못했습니다. 텍스트가 있는 작은 PDF로 다시 시도해 주세요.')
    if 'error' in result:
        raise DocumentError(result['error'])
    name = name.replace('\\', '/').rsplit('/', 1)[-1][:180] or 'document.pdf'
    return Document(name, result['pages'], make_chunks(result['pages'], tok), result['warnings'])
