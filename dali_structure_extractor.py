import os
import gzip
import pickle

def extract_hierarchical_lyrics(annot):
    """
    Extract nested paragraph-line-word-note structure from either 'hierarchical' or flat structures.
    Returns: List[List[List[str]]], where each entry is:
        - paragraph -> list of lines
        - line -> list of words
        - word -> list of note tokens
    """

    def from_hierarchical(hier_data):
        result = []
        for paragraph in hier_data:
            para_lines = []
            for line in paragraph.get("text", []):
                line_words = []
                for word in line.get("text", []):
                    notes = [note["text"].strip() for note in word.get("text", [])
                             if isinstance(note, dict) and "text" in note]
                    if notes:
                        line_words.append(notes)
                if line_words:
                    para_lines.append(line_words)
            if para_lines:
                result.append(para_lines)
        return result

    def from_flat_structure(annot):
        note_dict = {}
        for note in annot.get("notes", []):
            pid = note.get("parent_id")
            if "text" in note:
                note_dict.setdefault(pid, []).append(note["text"].strip())

        word_dict = {}
        for word in annot.get("words", []):
            wid = word.get("id")
            pid = word.get("parent_id")
            notes = note_dict.get(wid, [])
            if notes:
                word_dict.setdefault(pid, []).append(notes)

        line_dict = {}
        for line in annot.get("lines", []):
            lid = line.get("id")
            pid = line.get("parent_id")
            words = word_dict.get(lid, [])
            if words:
                line_dict.setdefault(pid, []).append(words)

        para_list = []
        for para in annot.get("paragraphs", []):
            pid = para.get("id")
            lines = line_dict.get(pid, [])
            if lines:
                para_list.append(lines)

        return para_list

    if "hierarchical" in annot:
        return from_hierarchical(annot["hierarchical"])
    elif all(k in annot for k in ("paragraphs", "lines", "words", "notes")):
        return from_flat_structure(annot)
    else:
        print("⚠️ Unknown annotation structure keys:", list(annot.keys()))
        return []


def test_dali_file_structure(file_path):
    with gzip.open(file_path, "rb") as f:
        raw = pickle.load(f)
    if raw.is_horizontal:
        raw.horizontal2vertical()
    annot = raw.annotations.get("annot", {})
    nested_lyrics = extract_hierarchical_lyrics(annot)
    for i, para in enumerate(nested_lyrics):
        print(f"\nParagraph {i+1}:")
        for j, line in enumerate(para):
            flat_line = [" ".join(word) for word in line]
            print(f"  Line {j+1}: {' | '.join(flat_line)}")
    return nested_lyrics
