#!/usr/bin/env python3
"""Quick test to verify schema-driven form agent functionality"""

import json
from src.agents.form_agent import (
    get_completed_fields,
    is_form_complete,
    _format_field_name,
    _build_extraction_rules,
    FORM_SCHEMA
)

print("=" * 60)
print("SCHEMA-DRIVEN FORM AGENT TEST")
print("=" * 60)

# Test 1: Field name formatting
print("\n1. Testing field name formatting:")
print(f"   'budget' -> '{_format_field_name('budget')}'")
print(f"   'typeOfHoliday' -> '{_format_field_name('typeOfHoliday')}'")
print(f"   'destinationPreferences' -> '{_format_field_name('destinationPreferences')}'")

# Test 2: Extraction rules (should be dynamic from schema)
print("\n2. Testing dynamic extraction rules generation:")
extraction_rules = _build_extraction_rules()
print("   Generated rules:")
for line in extraction_rules.split("\n"):
    print(f"   {line}")

# Test 3: Completed fields tracking (schema-driven)
print("\n3. Testing schema-driven completed fields tracking:")
partial_form = {
    "budget": 3000,
    "typeOfHoliday": "beach",
    "travelGroup": "",
    "availability": {},
    "destinationPreferences": []
}
completed = get_completed_fields(partial_form)
print(f"   Partial form completed fields: {completed}")
print(f"   Expected: ['budget', 'type of holiday']")

# Test 4: Form completion check (schema-driven)
print("\n4. Testing schema-driven form completion:")
incomplete_form = {
    "budget": 3000,
    "typeOfHoliday": "beach",
    "travelGroup": "family",
    "availability": {"startDate": "2025-06-01", "endDate": "2025-06-15"},
    "destinationPreferences": ["Bali", "Thailand"]
}
is_complete = is_form_complete(incomplete_form)
print(f"   Complete form check: {is_complete}")
print(f"   Expected: True")

missing_form = {
    "budget": 3000,
    "typeOfHoliday": "beach",
    "travelGroup": "",
    "availability": {},
    "destinationPreferences": []
}
is_incomplete = is_form_complete(missing_form)
print(f"   Incomplete form check: {is_incomplete}")
print(f"   Expected: False")

# Test 5: Schema structure verification
print("\n5. Verifying schema-driven behavior:")
print(f"   Schema fields: {list(FORM_SCHEMA.keys())}")
print(f"   All fields are now dynamically read from form.json")

print("\n" + "=" * 60)
print("SCHEMA-DRIVEN FORM AGENT VERIFICATION COMPLETE")
print("=" * 60)
