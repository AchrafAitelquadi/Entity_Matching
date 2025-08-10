import re
import os
import random
import numpy as np
import pandas as pd
from faker import Faker
faker = Faker('fr_FR')

cities = ['Casablanca', 'Rabat', 'Marrakech', 'Agadir', 'Tanger', 'Oujda', 'Kenitra']

def random_cin():
    return faker.random_uppercase_letter() + faker.random_uppercase_letter() + str(faker.random_number(digits=6, fix_len=True))

def random_cnss():
    return str(faker.random_number(digits=8, fix_len=True))

def introduce_typos(text):
    """Introduce realistic typos in text"""
    if not text or pd.isna(text):
        return text
    
    text = str(text)
    
    # Common typo patterns
    typo_patterns = [
        # Character substitutions (keyboard proximity)
        ('a', 'e'), ('e', 'a'), ('i', 'y'), ('o', '0'), ('0', 'o'),
        ('u', 'i'), ('m', 'n'), ('n', 'm'), ('b', 'v'), ('v', 'b'),
        ('c', 'k'), ('k', 'c'), ('s', 'z'), ('z', 's'),
        # Double letters
        ('l', 'll'), ('s', 'ss'), ('t', 'tt'), ('n', 'nn'),
        # Missing letters (deletions)
        ('th', 't'), ('ch', 'c'), ('qu', 'q'), ('tion', 'ion'),
        # Transpositions
        ('er', 're'), ('le', 'el'), ('on', 'no'), ('it', 'ti'),
    ]
    
    # Apply random typos (30% chance)
    if random.random() < 0.3:
        # Choose random typo type
        typo_type = random.choice(['substitute', 'delete', 'insert', 'transpose'])
        
        if typo_type == 'substitute' and len(text) > 1:
            # Character substitution
            pos = random.randint(0, len(text) - 1)
            chars = list(text)
            old_char = chars[pos].lower()
            
            # Use common substitutions or random
            substitutions = {
                'a': 'e', 'e': 'a', 'i': 'y', 'o': '0', 'u': 'i',
                'm': 'n', 'n': 'm', 'b': 'v', 'v': 'b', 'c': 'k', 'k': 'c'
            }
            
            if old_char in substitutions:
                chars[pos] = substitutions[old_char]
            else:
                # Random nearby character
                nearby_chars = 'abcdefghijklmnopqrstuvwxyz'
                chars[pos] = random.choice(nearby_chars)
            
            text = ''.join(chars)
        
        elif typo_type == 'delete' and len(text) > 2:
            # Delete a character
            pos = random.randint(0, len(text) - 1)
            text = text[:pos] + text[pos+1:]
        
        elif typo_type == 'insert':
            # Insert a character
            pos = random.randint(0, len(text))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            text = text[:pos] + char + text[pos:]
        
        elif typo_type == 'transpose' and len(text) > 1:
            # Transpose two adjacent characters
            pos = random.randint(0, len(text) - 2)
            chars = list(text)
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            text = ''.join(chars)
    
    return text

def add_formatting_issues(text):
    """Add realistic formatting issues"""
    if not text or pd.isna(text):
        return text
    
    text = str(text)
    
    # Random formatting issues
    if random.random() < 0.2:
        # Extra spaces
        text = re.sub(r'\s+', '  ', text)
    
    if random.random() < 0.3:
        # Missing spaces
        text = text.replace(' ', '')
    
    if random.random() < 0.2:
        # Random capitalization
        if random.random() < 0.5:
            text = text.upper()
        else:
            text = text.lower()
    
    if random.random() < 0.1:
        # Add random punctuation
        text = text + random.choice(['.', ',', ';', '!'])
    
    return text

def corrupt_phone_number(phone):
    """Add realistic phone number corruption"""
    if not phone or pd.isna(phone):
        return phone
    
    phone = str(phone)
    
    # Common phone number issues
    issues = [
        lambda p: p.replace('+212', ''),  # Remove country code
        lambda p: p.replace(' ', ''),     # Remove spaces
        lambda p: p.replace('-', ''),     # Remove dashes
        lambda p: p.replace('(', '').replace(')', ''),  # Remove parentheses
        lambda p: '0' + p if not p.startswith('0') else p,  # Add leading zero
        lambda p: p[1:] if p.startswith('0') else p,  # Remove leading zero
        lambda p: p.replace('06', '6', 1),  # Remove leading 0 in mobile
        lambda p: p.replace('05', '5', 1),  # Remove leading 0 in landline
    ]
    
    # Apply 1-2 random issues
    num_issues = random.randint(1, 2)
    for _ in range(num_issues):
        issue = random.choice(issues)
        phone = issue(phone)
    
    # Add typos to phone numbers
    phone = introduce_typos(phone)
    
    return phone

def corrupt_email(email):
    """Add realistic email corruption"""
    if not email or pd.isna(email):
        return email
    
    email = str(email)
    
    # Common email issues
    if random.random() < 0.3:
        # Wrong domain
        email = email.replace('@gmail.com', '@gmial.com')
        email = email.replace('@yahoo.com', '@yaho.com')
        email = email.replace('@hotmail.com', '@hotmial.com')
    
    if random.random() < 0.2:
        # Missing @ or .
        if '@' in email:
            email = email.replace('@', '', 1)
        elif '.' in email:
            email = email.replace('.', '', 1)
    
    if random.random() < 0.2:
        # Extra characters
        email = email.replace('@', '@@', 1)
    
    # Add typos
    email = introduce_typos(email)
    
    return email

def corrupt_address(address):
    """Add realistic address corruption"""
    if not address or pd.isna(address):
        return address
    
    address = str(address)
    
    # Common address abbreviations and issues
    abbreviations = {
        'Avenue': 'Ave', 'Boulevard': 'Blvd', 'Street': 'St', 'Road': 'Rd',
        'Rue': 'R.', 'Avenue': 'Av.', 'Boulevard': 'Bd.', 'Place': 'Pl.',
        'Quartier': 'Q.', 'Résidence': 'Rés.', 'Immeuble': 'Imm.',
        'Appartement': 'Apt', 'Numéro': 'N°', 'Bis': 'B'
    }
    
    # Apply abbreviations
    if random.random() < 0.4:
        for full, abbr in abbreviations.items():
            if full in address:
                address = address.replace(full, abbr)
    
    # Add typos
    address = introduce_typos(address)
    
    # Add formatting issues
    address = add_formatting_issues(address)
    
    return address

def add_data_entry_errors(person):
    """Add realistic data entry errors"""
    modified = person.copy()
    
    # Name issues
    if random.random() < 0.4:
        modified["full_name"] = introduce_typos(person["full_name"])
    
    if random.random() < 0.3:
        modified["full_name"] = add_formatting_issues(person["full_name"])
    
    # Email corruption
    if random.random() < 0.3:
        modified["email"] = corrupt_email(person["email"])
    
    # Phone corruption
    if random.random() < 0.4:
        modified["phone"] = corrupt_phone_number(person["phone"])
    
    # Address corruption
    if random.random() < 0.4:
        modified["address"] = corrupt_address(person["address"])
    
    # Employer name issues
    if random.random() < 0.3:
        modified["employer_name"] = introduce_typos(person["employer_name"])
    
    # CIN format variations
    if random.random() < 0.2:
        cin = person["cin"]
        if cin:
            # Remove letters or add spaces
            if random.random() < 0.5:
                modified["cin"] = cin.replace(cin[:2], cin[:2].lower())
            else:
                modified["cin"] = cin[:2] + " " + cin[2:]
    
    # CNSS number issues
    if random.random() < 0.2:
        cnss = person["cnss_number"]
        if cnss:
            # Add dashes or spaces
            if len(cnss) == 8:
                modified["cnss_number"] = cnss[:4] + "-" + cnss[4:]
    
    # Date format variations
    if random.random() < 0.2:
        dob = person["date_of_birth"]
        if dob:
            # Change date format
            from datetime import datetime
            try:
                date_obj = datetime.fromisoformat(dob)
                formats = [
                    "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"
                ]
                new_format = random.choice(formats)
                modified["date_of_birth"] = date_obj.strftime(new_format)
            except:
                pass
    
    return modified

def generate_person():
    return {
        "full_name": faker.name(),
        "cin": random_cin(),
        "date_of_birth": faker.date_of_birth(minimum_age=18, maximum_age=65).isoformat(),
        "place_of_birth": random.choice(cities),
        "cnss_number": random_cnss(),
        "email": faker.email(),
        "phone": faker.phone_number(),
        "address": faker.street_address(),
        "city": random.choice(cities),
        "employer_name": faker.company()
    }

def add_light_noise(person):
    """Add light noise to reference table"""
    modified = person.copy()
    
    # Light modifications (20% chance each)
    if random.random() < 0.2:
        modified["full_name"] = person["full_name"].title()  # case change
    
    if random.random() < 0.1:
        modified["email"] = person["email"].lower()
    
    if random.random() < 0.1:
        modified["phone"] = person["phone"].replace(" ", "")  # remove spaces
    
    return modified

def add_heavy_noise(person):
    """Add heavy noise to dataset table with realistic issues"""
    modified = person.copy()
    
    # Apply data entry errors first
    modified = add_data_entry_errors(modified)
    
    # Additional heavy modifications
    if random.random() < 0.6:
        modified["full_name"] = modified["full_name"].lower()  # case change
    
    if random.random() < 0.4:
        modified["address"] = modified["address"].replace("Rue", "R.")  # abbreviation
    
    if random.random() < 0.4:
        employer = modified["employer_name"]
        if employer and len(employer.split()) > 1:
            modified["employer_name"] = employer.split()[0]  # first word only
    
    # Add missing values (realistic missing data patterns)
    missing_probabilities = {
        'email': 0.15,      # Email often missing
        'phone': 0.1,       # Phone sometimes missing
        'address': 0.2,     # Address frequently incomplete
        'employer_name': 0.25,  # Employer often missing
        'cnss_number': 0.3,  # CNSS often missing in informal sectors
    }
    
    for field, prob in missing_probabilities.items():
        if random.random() < prob:
            modified[field] = None
    
    # Partial data corruption (common in real datasets)
    if random.random() < 0.1:
        # Truncated fields
        if modified["full_name"] and len(modified["full_name"]) > 10:
            modified["full_name"] = modified["full_name"][:10] + "..."
    
    if random.random() < 0.05:
        # Wrong data type (numbers as strings, etc.)
        if modified["phone"]:
            modified["phone"] = "TEL: " + str(modified["phone"])
    
    return modified

def create_similar_person(base_person):
    """Create a person that looks similar but is different (hard negative)"""
    similar = generate_person()
    
    # Keep some similar characteristics to make it tricky
    if random.random() < 0.3:
        # Same city
        similar["city"] = base_person["city"]
        similar["place_of_birth"] = base_person["place_of_birth"]
    
    if random.random() < 0.2:
        # Similar name (same first name or last name)
        base_name_parts = base_person["full_name"].split()
        similar_name_parts = similar["full_name"].split()
        if len(base_name_parts) >= 2 and len(similar_name_parts) >= 2:
            # Same first name, different last name
            similar["full_name"] = base_name_parts[0] + " " + similar_name_parts[-1]
    
    if random.random() < 0.1:
        # Similar employer
        similar["employer_name"] = base_person["employer_name"]
    
    if random.random() < 0.05:
        # Similar phone prefix (same area code)
        if base_person["phone"] and similar["phone"]:
            base_prefix = base_person["phone"][:3]
            similar["phone"] = base_prefix + similar["phone"][3:]
    
    return similar

def generate_duplicate_variations(person):
    """Generate multiple variations of the same person (realistic duplicates)"""
    variations = []
    
    # Create 2-4 variations
    num_variations = random.randint(2, 4)
    
    for _ in range(num_variations):
        variation = person.copy()
        
        # Apply different levels of corruption
        corruption_level = random.choice(['light', 'medium', 'heavy'])
        
        if corruption_level == 'light':
            variation = add_light_noise(variation)
        elif corruption_level == 'medium':
            variation = add_data_entry_errors(variation)
        else:  # heavy
            variation = add_heavy_noise(variation)
        
        variations.append(variation)
    
    return variations

def generate_tables(base_path, n_total=2000, match_ratio=0.3):
    """Generate reference table (cleaner) and source table (noisy)"""
    n_matches = int(n_total * match_ratio)
    n_non_matches = n_total - n_matches

    print(f"Generating {n_total} records:")
    print(f"{n_matches} matching pairs ({match_ratio*100:.0f}%)")
    print(f"{n_non_matches} non-matching records")
    print("=" * 50)

    # Create folders
    os.makedirs(os.path.join(base_path, "data"), exist_ok=True)

    # Generate base persons
    base_persons = [generate_person() for _ in range(n_matches)]

    # Reference table: light noise
    reference_table = []
    for i, person in enumerate(base_persons):
        ref_person = add_light_noise(person)
        ref_person['id'] = f"REF_{i:04d}"
        reference_table.append(ref_person)

    # Source table: heavy noise + hard negatives
    source_table = []
    for i, person in enumerate(base_persons):
        src_person = add_heavy_noise(person)
        src_person['id'] = f"SRC_{i:04d}"
        source_table.append(src_person)

    for i in range(n_matches, n_total):
        if random.random() < 0.3:
            base_for_similar = random.choice(base_persons)
            person = create_similar_person(base_for_similar)
        else:
            person = generate_person()

        if random.random() < 0.5:
            person = add_heavy_noise(person)

        person['id'] = f"SRC_{i:04d}"
        source_table.append(person)

    # Ground truth
    ground_truth = []
    for i in range(n_matches):
        ground_truth.append({
            'ref_id': f"REF_{i:04d}",
            'data_id': f"SRC_{i:04d}",
            'match': 1
        })

    for _ in range(n_matches * 2):
        ref_idx = random.randint(0, len(reference_table) - 1)
        data_idx = random.randint(0, len(source_table) - 1)
        ref_id = reference_table[ref_idx]['id']
        data_id = source_table[data_idx]['id']

        if not any(gt['ref_id'] == ref_id and gt['data_id'] == data_id for gt in ground_truth):
            ground_truth.append({'ref_id': ref_id, 'data_id': data_id, 'match': 0})

    # Shuffle
    random.shuffle(reference_table)
    random.shuffle(source_table)

    # Save
    pd.DataFrame(reference_table).to_csv(os.path.join(base_path, "data", "reference_table.csv"), index=False)
    pd.DataFrame(source_table).to_csv(os.path.join(base_path, "data", "source_table.csv"), index=False)
    pd.DataFrame(ground_truth).to_csv(os.path.join(base_path, "data", "ground_truth.csv"), index=False)

    # Print summary
    print("\n📁 Files saved")
    print(f" - Reference: {len(reference_table)} rows")
    print(f" - Source:    {len(source_table)} rows")
    print(f" - Ground Truth: {len(ground_truth)} pairs")
    print(f"   - Matches: {n_matches}")
    print(f"   - Non-matches: {len(ground_truth) - n_matches}")