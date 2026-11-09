# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""Build the PDF, synchronize AI citations, and package a completed submission.

Edit submission_details.json first. Run from the project root:
    python report/build_report.py
Optional dependencies: python -m pip install -r report/requirements.txt
"""
import html
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import fitz
import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
PDF = REPORT / "COSC_4117EL_A2_G1_Report.pdf"
ZIP = ROOT / "COSC_4117EL_A2_G1.zip"
CSS = """
body { font-family: sans-serif; font-size: 10.3pt; line-height: 1.35; color: #243242; }
h1 { font-size: 27pt; line-height: 1.15; color: #193d64; margin-bottom: 14pt; }
h2 { font-size: 16pt; line-height: 1.2; color: #193d64; margin: 16pt 0 10pt; }
h3 { font-size: 11.7pt; color: #193d64; margin: 13pt 0 7pt; }
p { margin: 7pt 0; }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; line-height: 1.25; margin: 10pt 0; }
th { color: #193d64; text-align: left; }
th, td { border: 0.5pt solid #c8d2dd; padding: 5pt; vertical-align: top; }
tr { page-break-inside: avoid; }
code { font-family: monospace; font-size: 8.5pt; }
pre { padding: 9pt; font-size: 9pt; }
li { margin-bottom: 5pt; }
em { color: #4b5a69; font-size: 9pt; }
a { color: #245e98; }
.pagebreak { page-break-before: always; }
.equation { padding: 10pt; margin: 10pt 0; text-align: center; }
.pending { color: #8a3e12; padding: 10pt; margin: 12pt 0; }
.missing { color: #a24411; font-weight: bold; }
"""


def display(value, label):
    if str(value).strip():
        return html.escape(str(value).strip())
    return '<span class="missing">[FILL: ' + html.escape(label) + ']</span>'


def prepare_details(details):
    missing = []
    if not details.get("group", "").strip():
        missing.append("group")
    members = details.get("members", [])
    if not members:
        missing.append("members")
    names = []
    rows = []
    has_contributions = any(member.get("contribution", "").strip() for member in members)
    for index, member in enumerate(members, 1):
        if not member.get("name", "").strip():
            missing.append(f"member {index}: name")
        name = display(member.get("name", ""), f"member {index} name")
        student_id = member.get("student_id", "").strip()
        if student_id:
            name += " (" + html.escape(student_id) + ")"
        names.append(name)
        contribution = html.escape(member.get("contribution", "").strip()) or "—"
        rows.append(f"<tr><td>{name}</td><td>{contribution}</td></tr>")
    links = details.get("ai_conversation_links", [])
    valid_links = []
    rendered_links = []
    for index, link in enumerate(links, 1):
        if not link.strip():
            continue
        url = urlparse(link.strip())
        if url.scheme in ("http", "https") and url.netloc:
            valid_links.append(link.strip())
            escaped = html.escape(link.strip(), quote=True)
            rendered_links.append(f'<a href="{escaped}">{escaped}</a>')
        else:
            missing.append(f"AI conversation link {index}")
            rendered_links.append(display("", f"AI conversation / prompt share link {index}"))
    repository = details.get("repository_url", "").strip()
    repository_link = ""
    if repository:
        url = urlparse(repository)
        if url.scheme in ("http", "https") and url.netloc:
            escaped = html.escape(repository, quote=True)
            repository_link = f'<p><b>Project repository:</b> <a href="{escaped}">{escaped}</a></p>'
        else:
            missing.append("repository URL")
    ai_links_section = ""
    if rendered_links:
        ai_links_section = "<p><b>Conversation / prompt share links:</b></p><ul>" + "".join("<li>" + link + "</li>" for link in rendered_links) + "</ul>"
        ai_links_section += "<p>The report builder includes these conversation links in the Python source comments.</p>"
    omission_note = ""
    if not valid_links and details.get("ai_link_omission_note", "").strip():
        omission_note = "<p><b>Conversation-link disclosure:</b> " + html.escape(details["ai_link_omission_note"].strip()) + "</p>"
    for key in ("ai_tools", "ai_use"):
        if not details.get(key, "").strip():
            missing.append(key)
    if details.get("d2l_programming_standards_checked") is not True:
        missing.append("D2L programming-standards review")
    group_details = "<ul>" + "".join("<li>" + name + "</li>" for name in names) + "</ul>"
    if has_contributions:
        group_details = '<table><tr><th>Member</th><th>Actual contribution</th></tr>' + "".join(rows) + "</table>"
    grade_statement = details.get("grade_adjustment_statement", "").strip()
    if grade_statement:
        grade_statement = "<p><b>Grade-adjustment statement:</b> " + html.escape(grade_statement) + "</p>"
    if details.get("d2l_programming_standards_checked") is True:
        standards_status = "The group has confirmed compliance with the separate D2L programming standards. Automated verification did not independently inspect that document."
    else:
        standards_status = "The separate D2L programming-standards document was not supplied, so compliance with it remains a group review item in SUBMISSION_CHECKLIST.md."
    values = {
        "GROUP": display(details.get("group", ""), "group name / number"),
        "MEMBERS": "; ".join(names) or display("", "group members"),
        "GROUP_DETAILS": group_details,
        "GRADE_STATEMENT": grade_statement,
        "STANDARDS_STATUS": standards_status,
        "AI_TOOLS": display(details.get("ai_tools", ""), "AI tools used"),
        "AI_USE": display(details.get("ai_use", ""), "description of AI use"),
        "AI_LINKS_SECTION": ai_links_section,
        "AI_LINK_OMISSION_NOTE": omission_note,
        "REPOSITORY_LINK": repository_link,
        "READINESS_NOTE": "",
    }
    if missing:
        values["READINESS_NOTE"] = '<div class="pending"><b>Submission details pending.</b> Remaining: ' + html.escape(", ".join(missing)) + '. Update submission_details.json and rebuild before uploading.</div>'
    return values, valid_links, missing


def update_code_citations(links, incomplete, repository="", omission_note=""):
    start = "# BEGIN AI ACKNOWLEDGMENT"
    end = "# END AI ACKNOWLEDGMENT"
    lines = [start, "# AI-assisted project preparation: OpenAI ChatGPT / Codex.",
             "# Scope and disclosure: report/REPORT.md, Section 10.",
             "# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C)."]
    if repository:
        lines.append("# Project repository: " + repository)
    lines.extend("# Conversation: " + link for link in links)
    if not links and omission_note:
        lines.append("# Conversation links omitted for privacy; see the disclosure in Section 10.")
    if incomplete:
        lines.append("# [FILL: add all actual AI share links in report/submission_details.json and rebuild.]")
    lines.append(end)
    block = "\n".join(lines) + "\n"
    for path in sorted(ROOT.glob("*.py")) + sorted(REPORT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        pattern = re.escape(start) + r"\n.*?" + re.escape(end) + r"\n?"
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL)
        if text.startswith("# -*- coding:"):
            first, remainder = text.split("\n", 1)
            text = first + "\n" + block + remainder
        else:
            text = block + text
        path.write_text(text, encoding="utf-8")


def build_pdf(source, values, details):
    for key, value in values.items():
        source = source.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", source):
        raise ValueError("Unresolved report template field")
    body = markdown.markdown(source, extensions=["tables", "fenced_code"])
    story = fitz.Story(body, user_css=CSS, archive=fitz.Archive(str(REPORT)))

    def page_rect(number, filled):
        return fitz.paper_rect("a4"), fitz.Rect(48, 44, 547, 790), None

    document = story.write_with_links(page_rect)
    for index, page in enumerate(document):
        if index:
            page.insert_text((48, 26), "COSC-4117EL | Assignment 2 | Warehouse Robot", fontsize=8, color=(.34, .40, .47))
        page.draw_line((48, 804), (547, 804), color=(.76, .81, .86), width=.5)
        page.insert_text((48, 819), "Dynamic Programming and Reinforcement Learning", fontsize=8, color=(.34, .40, .47))
        page.insert_text((480, 819), f"{index + 1} / {len(document)}", fontsize=8, color=(.34, .40, .47))
    toc = []
    for title in re.findall(r"^## (.+)$", source, flags=re.MULTILINE):
        if not re.match(r"\d+\.|Appendix", title):
            continue
        for index, page in enumerate(document):
            heading_parts = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["size"] >= 15:
                            heading_parts.append(span["text"])
            headings = unicodedata.normalize("NFKC", " ".join(heading_parts))
            if unicodedata.normalize("NFKC", title) in headings:
                toc.append([1, title, index + 1])
                break
    document.set_toc(toc)
    document.set_metadata({"title": "Warehouse Robot Decision Making — COSC-4117EL Assignment 2",
                           "author": ", ".join(m.get("name", "") for m in details.get("members", []) if m.get("name")),
                           "subject": "MDP formulation, Value Iteration, Q-Learning, and stochastic motion",
                           "creator": "report/build_report.py (Markdown and PyMuPDF)"})
    document.save(PDF, garbage=4, deflate=True)
    page_count = len(document)
    document.close()
    return page_count


def package_submission():
    files = list(ROOT.glob("*.py"))
    files += [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "SUBMISSION_CHECKLIST.md"]
    files += list((ROOT / "outputs").glob("*.csv"))
    for pattern in ("*.py", "*.md", "*.json", "*.txt", "*.pdf", "figures/*.png"):
        files.extend(REPORT.glob(pattern))
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(ZIP) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Submission ZIP integrity check failed")


def main():
    details = json.loads((REPORT / "submission_details.json").read_text(encoding="utf-8"))
    values, links, missing = prepare_details(details)
    update_code_citations(links, any(item.startswith("AI conversation") for item in missing),
                          details.get("repository_url", "").strip(),
                          details.get("ai_link_omission_note", "").strip())
    values["AI_ASSISTANCE_REPORT"] = (REPORT / "AI_ASSISTANCE.md").read_text(encoding="utf-8")
    pages = build_pdf((REPORT / "REPORT.md").read_text(encoding="utf-8"), values, details)
    print(f"Built {PDF.relative_to(ROOT)} ({pages} pages)")
    if missing:
        print("Remaining fields in report/submission_details.json:")
        for item in missing:
            print(" -", item)
        print("Final ZIP not created: finish these details, then run this builder again.")
        if ZIP.exists():
            print("An older ZIP exists; it was not updated. Do not submit a stale ZIP.")
    else:
        package_submission()
        print(f"Created {ZIP.name} ({ZIP.stat().st_size / 1024 / 1024:.2f} MiB)")
    if not links:
        print("AI conversation links are omitted. The repository link does not replace the assignment's prompt-link requirement.")


if __name__ == "__main__":
    main()
