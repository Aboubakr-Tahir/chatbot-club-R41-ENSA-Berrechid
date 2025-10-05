from datetime import datetime

# Get current date
now = datetime.now()

# Format the current date as "Month Day, Year" (e.g., "October 5, 2025")
current_date_str = now.strftime("%B %d, %Y")

# Determine academic year: if month is before September, academic year started last year
if now.month < 9:
    academic_year_start = now.year - 1
else:
    academic_year_start = now.year
current_academic_year = f"{academic_year_start}-{academic_year_start + 1}"
previous_academic_year = f"{academic_year_start - 1}-{academic_year_start}"

# Build the dynamic string
dynamic_string = f"""The current date is {current_date_str}.
The current academic year is {current_academic_year}.
The previous academic year was {previous_academic_year}."""

print(dynamic_string)