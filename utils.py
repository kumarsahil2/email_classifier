import re

# Updated PII Patterns with strict digit counts and flexible separators
PII_PATTERNS = {
    "email": r"\b[\w\.-]+?@\w+?\.\w{2,4}\b",
    "phone_number": r"\+91[\s\-]*((?:\d[\s\-]*){10})\b",
    "dob": r"\b(?:\d{2}[\/\-.]){2}\d{4}\b",
    "aadhar_num": r"\b(?:\d[\s\-]*){12}\b",
    "credit_debit_no": r"\b(?:\d[\s\-]*){16}\b",
    "cvv_no": r"(?i)\bCVV[:\s\-]*\d{3}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/\d{2,4}\b",
    "full_name": r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"
}

def mask_pii(email_text):
    entities = []
    masked_text = email_text
    already_masked_positions = []

    # Sort patterns to avoid masking overlaps (longer entities first)
    sorted_patterns = sorted(PII_PATTERNS.items(), key=lambda x: -len(x[1]))

    for entity_type, pattern in sorted_patterns:
        for match in re.finditer(pattern, masked_text):
            start, end = match.start(), match.end()

            # Skip overlapping matches
            if any(s <= start < e or s < end <= e for s, e in already_masked_positions):
                continue

            original = masked_text[start:end]
            replacement = f"[{entity_type}]"
            masked_text = masked_text[:start] + replacement + masked_text[end:]

            # Recalculate for future skips
            already_masked_positions.append((start, start + len(replacement)))

            entities.append({
                "position": [start, start + len(replacement)],
                "classification": entity_type,
                "entity": original
            })

    return masked_text, entities
