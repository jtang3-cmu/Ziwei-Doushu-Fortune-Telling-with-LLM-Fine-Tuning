import os
import json
import re
from pathlib import Path

# path to the books
BOOK_FILES = [
    "ziwei_chart_rules.txt",
]

# output path
OUTPUT_DIR = "rag_corpus"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rag.jsonl")

# chunk size setting
MAX_CHARS = 1600   # maximum length for each chunk 
MIN_CHARS = 1000   # minimum length for each chunk


# ---------- Process chart content ----------

def looks_like_separator(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if set(line) <= set("-—+|= "):
        return True
    if "----" in line:
        return True
    return False


def is_chart_line(line: str) -> bool:
    """
    recognize the paragraph is chart or not
    """
    s = line.strip()
    if not s:
        return False

    if looks_like_separator(s):
        return True

    if s.count("|") >= 2:
        return True

    # typical words than would appear in the chart
    CHART_CHARS = "一二三四五六七八九十初廿卅子丑寅卯辰巳午未申酉戌亥"
    if any(c in CHART_CHARS for c in s):
        if not any(p in s for p in "。！？；：，、"):
            return True

    return False


# ---------- split sentences and group into chunk ----------

def split_text_into_chunks(text: str,
                           max_chars: int = MAX_CHARS,
                           min_chars: int = MIN_CHARS) -> list[str]:
    """seperate using chinese period(。), and then group into chunk"""
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # split using (。|！|？)
    parts = re.split(r"(。|！|？)", text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if i + 1 < len(parts):
            sent += parts[i + 1]
        if sent.strip():
            sentences.append(sent)

    chunks = []
    buf = ""
    for sent in sentences:
        if looks_like_separator(sent):
            if buf.strip():
                chunks.append(buf.strip())
                buf = ""
            chunks.append(sent.strip())
            continue

        if len(buf) + len(sent) > max_chars and len(buf) >= min_chars:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf += sent

    if buf.strip():
        chunks.append(buf.strip())

    return chunks


# ---------- Parse manually labelled text ----------

def parse_marked_book(path: str):
    """
    rule: each topic should start with "###"
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"{path} not found")

    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks = []
    big_title = filepath.stem 

    buf = []

    def flush_buf():
        nonlocal buf
        if not buf:
            return
        text = "".join(buf).strip()
        if not text:
            buf = []
            return
        blocks.append({
            "source_file": str(filepath.name),
            "big_title": big_title,
            "mid_title": None,
            "small_title": None,
            "block_type": "text",
            "text": text,
        })
        buf = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("###"):
            # save last paragraph into flush
            flush_buf()
            # remove marks
            title = stripped[3:].lstrip()
            if title:
                buf.append(title + "\n")
        else:
            buf.append(line)

    flush_buf()

    return blocks





# ---------- main function: building docs.jsonl ----------

def build_corpus():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for file_path in BOOK_FILES:
            if not os.path.exists(file_path):
                print(f"[WARN] {file_path} not found, skip.")
                continue

            print(f"Parsing {file_path} ...")
            blocks = parse_marked_book(file_path)
            print(f"  -> {len(blocks)} blocks")

            for block_idx, block in enumerate(blocks):
                text = block["text"]

                # form each block into one chunk
                chunk_text = text.strip()
                if not chunk_text:
                    continue

                doc_id = f"{Path(block['source_file']).stem}-blk{block_idx}-chunk0"

                doc = {
                    # "id": doc_id,
                    "big_title": block["big_title"],
                    # "mid_title": block["mid_title"],    
                    # "small_title": block["small_title"],
                    "block_type": block["block_type"], 
                    # "chunk_index": 0,
                    # "source_file": block["source_file"],
                    "text": chunk_text,
                }
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                doc_count += 1

    print(f"Done. Wrote {doc_count} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_corpus()
