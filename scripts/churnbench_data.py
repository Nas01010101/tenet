"""Static generator data for scripts/bench_churn.py — split out to keep bench_churn.py
under the repo's 500-line file limit. No logic here, just attribute pools + templates.

Attribute pools: each >=40 unique values so a chain can go up to U=32 without repeats.
Cross-checked (see test_churnbench.py) that no value in a pool is a substring of another
value in the SAME pool, so substring scoring can't be fooled by a stale value accidentally
containing/being-contained-in the current one.
"""
from __future__ import annotations

_CITIES = [
    "Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mexico City", "Cairo", "Mumbai", "Beijing",
    "Dhaka", "Osaka", "Karachi", "Chongqing", "Istanbul", "Buenos Aires", "Kolkata", "Lagos",
    "Kinshasa", "Manila", "Tianjin", "Guangzhou", "Lahore", "Moscow", "Shenzhen", "Bangalore",
    "Paris", "Bogota", "Jakarta", "Chennai", "Lima", "Bangkok", "Seoul", "Nagoya", "Hyderabad",
    "London", "Tehran", "Chicago", "Chengdu", "Nanjing", "Wuhan", "Toronto",
]
_LEVELS = ["junior", "associate", "senior", "staff", "principal", "lead", "head", "chief"]
_ROLES = ["analyst", "engineer", "designer", "strategist", "consultant"]
_JOBS = [f"{lvl} {role}" for role in _ROLES for lvl in _LEVELS]  # 40, level-first token unique
_CARS = [
    "Honda Civic", "Toyota Camry", "Tesla Model 3", "Ford F-150", "BMW 3 Series", "Audi A4",
    "Mazda CX-5", "Subaru Outback", "Chevrolet Malibu", "Nissan Altima", "Hyundai Elantra",
    "Kia Sportage", "Volkswagen Golf", "Jeep Wrangler", "Ram 1500", "GMC Sierra",
    "Chrysler Pacifica", "Dodge Charger", "Lexus RX", "Acura TLX", "Infiniti Q50",
    "Volvo XC60", "Porsche 911", "Mercedes C-Class", "Cadillac Escalade", "Buick Enclave",
    "Lincoln Navigator", "Mitsubishi Outlander", "Genesis G70", "Alfa Romeo Giulia",
    "Mini Cooper", "Fiat 500", "Land Rover Defender", "Jaguar F-Pace", "Chevrolet Bolt",
    "Ford Mustang", "Toyota Prius", "Honda Accord", "Mazda3", "Subaru Forester",
]
_PHONE_BRANDS = ["iPhone", "Galaxy S", "Pixel", "OnePlus", "Xperia"]
_PHONE_NUMS = [str(n) for n in range(10, 18)]  # two-digit only, avoids "iPhone 1" substring issues
_PHONES = [f"{b} {n}" for b in _PHONE_BRANDS for n in _PHONE_NUMS]  # 40
_GYM_BASES = ["Planet Fitness", "Equinox", "CrossFit Central", "Gold's Gym", "LA Fitness",
              "Anytime Fitness", "Life Time", "Blink Fitness"]
_GYM_LOCS = ["Downtown", "Uptown", "Westside", "Eastside", "Midtown"]
_GYMS = [f"{b} {loc}" for b in _GYM_BASES for loc in _GYM_LOCS]  # 40

ATTR_SPECS = {
    "residence": {"question": "Which city does the user currently live in?",
                  "update": "I just moved to {v}.", "pool": _CITIES},
    "job_title": {"question": "What is the user's current job title?",
                  "update": "I got promoted, I'm now a {v}.", "pool": _JOBS},
    "car": {"question": "What car does the user currently drive?",
            "update": "I bought a new car, a {v}.", "pool": _CARS},
    "phone": {"question": "What phone does the user currently use?",
              "update": "I switched phones, now using a {v}.", "pool": _PHONES},
    "gym": {"question": "Which gym does the user currently go to?",
            "update": "I changed gyms, I go to {v} now.", "pool": _GYMS},
}
ATTR_ORDER = list(ATTR_SPECS)  # deterministic n_facts<len(ATTR_SPECS) subset selection

DISTRACTORS = [
    "The weather has been really nice this week.", "I watched a great documentary last night.",
    "I'm thinking about learning to cook Thai food.", "Traffic was terrible this morning.",
    "I read an interesting article about space travel.", "My neighbor got a new dog, it's adorable.",
    "I've been trying to drink more water lately.", "The coffee shop downtown changed its hours.",
    "I started a new book last night.", "Work has been busy lately.",
    "I need to renew my passport soon.", "I tried a new recipe yesterday.",
    "The gym was crowded today.", "I've been meaning to organize my closet.",
    "My phone battery has been draining fast.", "I saw a great concert last month.",
    "The bus was late again this morning.", "I'm planning a trip for the holidays.",
    "I've been listening to a lot of podcasts.", "My plants finally started blooming.",
    "I need to get my car's oil changed.", "There's construction on my street.",
    "I picked up a new hobby recently.", "The new season of my favorite show dropped.",
]
CHUNK_SIZE = 8  # turns per simulated ingest_session chunk
