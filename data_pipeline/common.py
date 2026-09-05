"""Shared helpers for MoSPI report parsing."""
import re

# Sectors used in MoSPI flash reports / QPISR (canonical spellings, upper)
FLASH_SECTORS = [
    "ATOMIC ENERGY", "CIVIL AVIATION", "COAL", "COMMUNICATIONS", "DEFENCE PRODUCTION",
    "DEVELOPMENT OF NORTH EASTERN REGION", "DONER", "FAMILY WELFARE", "FINANCE",
    "HEALTH", "HIGHER EDUCATION", "HOUSING", "MINES", "PETROLEUM", "PORTS AND SHIPPING",
    "POWER", "RAILWAYS", "RENEWABLE ENERGY", "ROAD TRANSPORT AND HIGHWAYS", "STEEL",
    "TELECOMMUNICATIONS", "URBAN DEVELOPMENT", "WATER RESOURCES",
    "ROAD TRANSPORT", "AND HIGHWAYS", "URBAN", "DEVELOPMENT",
]

STATES = [
    "ANDAMAN AND NICOBAR ISLANDS", "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM",
    "BIHAR", "CHHATTISGARH", "DELHI", "GOA", "GUJARAT", "HARYANA",
    "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA",
    "LADAKH", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM",
    "MULTIPLE STATES", "NAGALAND", "NEW DELHI", "ODISHA", "PUDUCHERRY", "PUNJAB",
    "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TELANGANA", "TRIPURA", "UTTAR PRADESH",
    "UTTARAKHAND", "WEST BENGAL", "DADRA AND NAGAR HAVELI", "TELENGANA",
    "ANDAMAN AND", "JAMMU AND", "JAMMU AND KASHMIR AND LADAKH", "NOT APPLICABLE",
]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_month_year(tok: str):
    """Parse 'MM/YYYY', 'M/YYYY', 'MM-YYYY', 'May-23', 'May-2023', 'May 2023' -> (yyyy, mm) or None."""
    if not tok:
        return None
    tok = tok.strip().strip('.,;')
    m = re.match(r'^(\d{1,2})[/\-.](\d{4})$', tok)
    if m:
        mm, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12:
            return (yy, mm)
        return None
    m = re.match(r'^([A-Za-z]{3,})[\-\s/]?(\d{2,4})$', tok)
    if m:
        mon = m.group(1)[:3].upper()
        if mon in MONTHS:
            yy = int(m.group(2))
            if yy < 100:
                yy += 2000
            return (yy, MONTHS[mon])
    return None


def parse_number(tok: str):
    """Parse '1,405.00', '1405', '-', 'N.A.', 'NA' -> float or None."""
    if tok is None:
        return None
    t = tok.strip().replace(',', '').replace('`', '')
    if t in ('-', '--', '', 'N.A.', 'NA', 'N.A', 'Nil', 'NIL', 'nan'):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def month_index(ym):
    if not ym:
        return None
    return ym[0] * 12 + (ym[1] - 1)


def months_between(a, b):
    """Months from a to b (b - a). Positive if b later."""
    ia, ib = month_index(a), month_index(b)
    if ia is None or ib is None:
        return None
    return ib - ia


def norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())
