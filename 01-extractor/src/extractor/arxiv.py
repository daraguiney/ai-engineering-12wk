import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
TIMEOUT_SECONDS = 30


def parse_atom(xml_bytes: bytes, arxiv_id: str) -> str:
    """Turn an arXiv Atom response into plain text for the extractor."""
    entry = ET.fromstring(xml_bytes).find(f"{ATOM}entry")
    if entry is None:
        raise ValueError(f"No arXiv entry for {arxiv_id}")
    title = " ".join(entry.findtext(f"{ATOM}title", "").split())
    authors = [a.findtext(f"{ATOM}name", "") for a in entry.findall(f"{ATOM}author")]
    summary = " ".join(entry.findtext(f"{ATOM}summary", "").split())
    return f"Title: {title}\nAuthors: {', '.join(authors)}\n\nAbstract: {summary}"


def fetch_arxiv_text(arxiv_id: str) -> str:
    url = ARXIV_API.format(id=arxiv_id)
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
        return parse_atom(resp.read(), arxiv_id)
