from pathlib import Path
from enum import Enum
import hashlib
import csv
import json
import docx
import pypdf

class FileType(Enum):
    DOCX = ".docx"
    PDF = ".pdf"
    CSV = ".csv"
    JSON = ".json"
    TXT = ".txt"
    MD = ".md"

def string_to_hash_id(source_file: str, source_type: str, block_id: int, text: str) -> str:
    key = f"{source_file}::{source_type}::{block_id}::{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def parse_document(file_path: str | Path) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()
    file_str = str(path)
    blocks: list[str] = []

    try:
        file_type = FileType(ext)
    except ValueError:
        raise ValueError(f"Unsupported format: {ext}")

    match file_type:
        # 1. DOCX
        case FileType.DOCX:
            doc = docx.Document(path)
            text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            if text:
                blocks.append(text)

        # 2. PDF
        case FileType.PDF:
            reader = pypdf.PdfReader(path)
            page_texts = [
                page.extract_text().strip()
                for page in reader.pages
                if page.extract_text() and page.extract_text().strip()
            ]
            full_text = "\n\n".join(page_texts)
            if full_text:
                blocks.append(full_text)

        # 3. CSV
        case FileType.CSV:
            with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)

                if header:
                    for row in reader:
                        if row:
                            paired_row = [
                                f"{h.strip()}: {v.strip()}"
                                for h, v in zip(header, row)
                            ]
                            blocks.append(" | ".join(paired_row))

        # 4. JSON
        case FileType.JSON:
            with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, list):
                    blocks = [json.dumps(item, ensure_ascii=False) for item in data]
                else:
                    blocks = [json.dumps(data, indent=2, ensure_ascii=False)]

        # 5. TXT / MD
        case FileType.TXT | FileType.MD:
            with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    blocks.append(content)

    output_blocks = []
    for i, block in enumerate(blocks):
        output_blocks.append(
            {
                "id": string_to_hash_id(file_str, ext, i, block),
                "source_file": file_str,
                "source_type": ext,
                "text": block,
            }
        )

    return output_blocks