"""
Generates the labeled training dataset for the triage classifier.

Rather than hand-typing 200+ sentences, we use templates with swappable
slots (names, locations, injury/resource details) so the dataset has real
lexical diversity instead of every 'trapped_injured' example looking
identical. Severity is drawn from a category-appropriate range so the
model learns that severity varies *within* a category too, not just
between categories (a "medical" message can be a 3 or a 5 depending on
wording) -- this matters for the demo since it shows real modeling, not
a lookup table.
"""

import random
import csv

random.seed(7)

NAMES = ["my father", "my mother", "my sister", "my neighbor", "an elderly man",
         "a child", "my grandmother", "two of my coworkers", "my husband",
         "the family next door", "a pregnant woman", "my roommate"]

PLACES = ["near the old bridge", "on Church Street", "behind the market",
          "at the school shelter", "on the second floor", "near the riverbank",
          "in the basement", "close to the highway", "by the water tank",
          "on the rooftop", "at the bus stand", "in the collapsed building"]

# Each category: list of (severity, template) pairs. Severity is tied to the
# specific wording of each template (not assigned randomly) so the model has
# an actual signal to learn -- "losing consciousness" should predict higher
# severity than "getting dehydrated", regardless of category.
#
# Deliberately varied vocabulary within each severity tier (not just
# name/place swapped into one sentence shape) -- this is what lets the
# model learn "unconscious / not breathing / unresponsive" all mean the
# same thing, instead of memorizing one exact phrase.
TEMPLATES = {
    "trapped_injured": [
        (5, "Trapped {place} for over an hour, {name} is losing consciousness"),
        (5, "Building collapsed {place}, {name} is stuck inside and not responding"),
        (5, "{name} is not breathing properly, pinned under something heavy {place}"),
        (5, "{name} is unconscious and buried under rubble {place}, please hurry"),
        (5, "We are crushed under a fallen wall {place}, {name} can't feel their legs"),
        (4, "{name} is trapped under debris {place}, please send rescue immediately"),
        (4, "We are pinned down by fallen concrete {place}, {name} is bleeding heavily"),
        (4, "{name} is badly injured and can't move, {place}, need help now"),
        (4, "{name} got stuck between fallen beams {place} and is in a lot of pain"),
        (3, "{name} is stuck {place} but conscious and talking, just can't get out"),
    ],
    "medical": [
        (5, "{name} has a high fever and won't wake up, we are {place}"),
        (5, "{name} is having a severe allergic reaction, {place}, please advise"),
        (5, "{name} collapsed suddenly and isn't responding to us, {place}"),
        (5, "{name} is having trouble breathing and lips are turning blue, {place}"),
        (4, "{name} is having severe chest pain {place}, no ambulance can reach us"),
        (4, "{name} just went into labor {place}, we need a medical team fast"),
        (4, "{name} is vomiting blood {place}, we don't know what's wrong"),
        (3, "{name} needs insulin urgently, we ran out and can't get to a pharmacy"),
        (3, "{name} has a deep cut that won't stop bleeding, {place}"),
        (3, "{name} is showing signs of infection from an untreated wound {place}"),
    ],
    "structural_rescue": [
        (5, "Wall collapsed onto the shelter {place}, unknown number of people underneath"),
        (5, "Entire floor gave way {place}, several people fell through, screaming heard"),
        (4, "Roof partially caved in {place}, {name} and others are trapped in the room below"),
        (4, "The building {place} is cracking and could collapse, people still inside"),
        (4, "A gas line ruptured near the collapsed structure {place}, people still trapped"),
        (3, "Bridge {place} is unstable, several vehicles stuck, need structural rescue team"),
        (3, "Balcony is hanging loose {place}, residents refusing to leave, needs assessment"),
        (3, "There is a large crack spreading across the wall {place} and it keeps getting worse"),
        (3, "Foundation looks shifted {place}, floor is uneven and door frames won't close"),
        (4, "A live wire fell {place}, nobody can safely get in or out until it's handled"),
        (4, "Gas smell is getting stronger {place}, worried the structure isn't safe anymore"),
    ],
    "water_food": [
        (4, "{name} needs clean water urgently, {place}, getting dehydrated"),
        (4, "Infant needs formula milk {place}, we have none left and store is shut"),
        (3, "Running low on drinking water {place}, six of us stranded here"),
        (3, "No food left for two days {place}, mostly children and elderly here"),
        (3, "We are starving {place}, haven't had a proper meal in three days"),
        (2, "We have supplies but can't reach the distribution point {place}"),
        (2, "Food stock is getting low {place}, should last one more day"),
    ],
    "evacuation": [
        (4, "Water is rising fast {place}, we need an evacuation boat soon"),
        (4, "Fire is spreading {place}, requesting evacuation transport"),
        (4, "House is filling up with smoke {place}, need immediate evacuation"),
        (3, "{name} needs to be evacuated {place}, road is flooded, no other way out"),
        (3, "Landslide warning issued near {place}, requesting transport out"),
        (2, "We are safe for now but stranded {place}, requesting evacuation route"),
        (2, "Considering evacuating {place} soon, would like guidance on safest route"),
    ],
    "informational": [
        (2, "Just checking if anyone has heard from {name}, no updates since morning"),
        (2, "Slightly worried since {name} hasn't replied since last night, otherwise fine"),
        (1, "Can someone confirm if the shelter {place} is still open"),
        (1, "We are safe {place}, just letting the family group know"),
        (1, "Is the road {place} clear yet, planning to head back home"),
        (1, "Does anyone know if power has been restored {place}"),
        (1, "Has anyone seen my dog, he ran off during the storm {place}"),
        (1, "Wanted to know if the main road towards the city is passable now"),
        (1, "Just wanted to say thank you to the volunteers helping {place}"),
    ],
}


def generate_dataset(rows_per_category: int = 35):
    rows = []
    for category, templates in TEMPLATES.items():
        for _ in range(rows_per_category):
            severity, template = random.choice(templates)
            text = template.format(name=random.choice(NAMES).capitalize() if template.startswith("{name}")
                                    else random.choice(NAMES),
                                    place=random.choice(PLACES))
            rows.append({"text": text, "category": category, "severity": severity})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate_dataset(rows_per_category=60)
    with open("data/sos_messages.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "severity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled examples to data/sos_messages.csv")
    print("\nSample:")
    for r in rows[:8]:
        print(f"  [{r['category']:18s} sev={r['severity']}] {r['text']}")
