"""Create The Tale of Peter Rabbit as a demo .cvn story file.

Public domain text by Beatrix Potter (1902).
Characters, chapters, pages, panels, and scripts — all fields populated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import Story, Character, ContentStore
from backend.models.storage import save_story
from backend.models.profile import Profile, PhysicalTraits, Outfit
from backend.models.appearance import AppearanceProperties


def create_peter_rabbit():
    story = Story("peter-rabbit", "The Tale of Peter Rabbit")
    story.synopsis = (
        "A naughty young rabbit disobeys his mother and sneaks into "
        "Mr. McGregor's garden, narrowly escaping with his life."
    )
    story.art_style = "storybook illustration, soft watercolor, warm tones"
    story.genre = "children's adventure"
    story.negative_prompt = "photo, realistic, 3d render, dark, horror"

    content_store = ContentStore("data/content")

    # === Characters ===

    peter = Character(
        "peter", "Peter Rabbit",
        description="A naughty, curious young rabbit who can't resist adventure.",
        personality_prompt="curious, naughty, impulsive, brave but easily frightened",
    )
    peter.appearance.properties = AppearanceProperties(
        species="rabbit",
        body_type="small, young, lean",
        height="short",
        skin_tone="brown fur",
        hair_style="",
        hair_colour="",
        eye_colour="bright brown",
        facial_features="twitchy nose, long upright ears, whiskers",
        outfit="blue jacket with brass buttons",
        accessories="",
        art_style_notes="Beatrix Potter style, soft watercolor",
    )
    peter.negative_prompt = "human, human ears, no fur, realistic"
    peter.profile.biography = (
        "The youngest and most mischievous of four rabbit siblings. "
        "His father was caught by Mr. McGregor and put in a pie."
    )
    peter.profile.physical = PhysicalTraits(
        body="small young rabbit, lean and quick",
        face="bright curious eyes, twitchy nose, long upright ears",
        distinguishing_marks="white cotton tail",
        hair_fur="soft brown fur",
    )
    peter.profile.add_outfit("blue jacket", "blue jacket with brass buttons, quite new", is_default=True)
    peter.profile.add_outfit("naked", "no clothes — lost his jacket in McGregor's garden")
    peter.profile.tendencies = [
        "gets into trouble", "runs headfirst into danger", "cries when scared"
    ]
    peter.profile.expressions = {
        "scared": "ears flat back, eyes wide, trembling",
        "naughty": "mischievous grin, ears perked forward",
        "sick": "droopy ears, half-closed eyes, hunched over",
        "crying": "big tears, ears drooping, sitting hunched",
    }
    story.add_character(peter)

    mrs_rabbit = Character(
        "mrs-rabbit", "Mrs. Rabbit",
        description="Peter's mother. Sensible, caring, warns her children about danger.",
        personality_prompt="caring, sensible, maternal, firm but loving",
    )
    mrs_rabbit.appearance.properties = AppearanceProperties(
        species="rabbit",
        body_type="plump, motherly",
        skin_tone="brown fur",
        eye_colour="kind brown",
        facial_features="gentle expression, neat whiskers",
        outfit="neat apron",
        accessories="basket, umbrella",
        art_style_notes="Beatrix Potter style, soft watercolor",
    )
    mrs_rabbit.negative_prompt = "human, human ears, no fur, realistic"
    mrs_rabbit.profile.physical = PhysicalTraits(
        body="plump motherly rabbit",
        face="kind eyes, gentle expression",
        hair_fur="brown fur, neat and well-groomed",
    )
    mrs_rabbit.profile.add_outfit("apron", "neat apron, carries a basket and umbrella", is_default=True)
    story.add_character(mrs_rabbit)

    mcgregor = Character(
        "mcgregor", "Mr. McGregor",
        description="A grumpy old gardener who chases rabbits out of his garden.",
        personality_prompt="grumpy, determined, territorial about his garden",
    )
    mcgregor.appearance.properties = AppearanceProperties(
        species="human",
        body_type="stocky, strong from gardening",
        height="tall",
        skin_tone="weathered, sun-tanned",
        hair_style="bald",
        eye_colour="dark",
        facial_features="bushy eyebrows, scowling, weathered face",
        outfit="dirty overalls, boots",
        accessories="straw hat, rake",
        art_style_notes="Beatrix Potter style, soft watercolor",
    )
    mcgregor.negative_prompt = "animal ears, tail, fur, snout, anthro"
    mcgregor.profile.physical = PhysicalTraits(
        body="stocky old man, strong from gardening",
        face="weathered face, bushy eyebrows, scowling",
    )
    mcgregor.profile.add_outfit("gardening", "dirty overalls, boots, straw hat", is_default=True)
    mcgregor.profile.tendencies = [
        "shakes his fist", "waves garden tools threateningly", "stomps around"
    ]
    story.add_character(mcgregor)

    # === Chapter 1: The Warning ===
    ch1 = story.create_chapter(
        "The Warning", ["peter", "mrs-rabbit"],
        synopsis="Mrs. Rabbit warns her children not to go into Mr. McGregor's garden.",
        default_location="rabbit burrow under a big fir tree",
        default_time_of_day="morning",
    )
    ch1.negative_prompt = "dark, night, scary"

    # Page 1: The family
    p1 = ch1.pages[0]
    p1.setting = "cozy rabbit burrow, warm morning light filtering through roots"
    p1.mood = "warm, domestic"
    p1.action_context = "morning routine"
    p1.time_of_day = "morning"
    p1.weather = "clear"
    p1.lighting = "warm sunlight"

    pan1 = p1.panels[0]
    pan1.narration = "Once upon a time there were four little Rabbits."
    pan1.shot_type = "medium"
    pan1.scripts["peter"].update(
        action="sitting with siblings",
        emotion="restless",
        pose="sitting cross-legged",
        outfit="blue jacket",
        source="manual",
    )
    pan1.scripts["mrs-rabbit"].update(
        dialogue="Now my dears, you may go into the fields, but don't go into Mr. McGregor's garden.",
        action="standing at burrow entrance",
        emotion="serious",
        pose="standing upright",
        outfit="apron",
        direction="medium shot, warm lighting",
        source="manual",
    )

    # Page 2: Peter sneaks off
    p2 = ch1.create_page()
    p2.setting = "garden gate, lush vegetable garden beyond the fence"
    p2.mood = "mischievous, exciting"
    p2.action_context = "sneaking"
    p2.time_of_day = "morning"
    p2.weather = "clear"
    p2.lighting = "bright sunlight"

    pan2 = p2.panels[0]
    pan2.narration = "But Peter, who was very naughty, ran straight away to Mr. McGregor's garden."
    pan2.shot_type = "low angle"
    pan2.scripts["peter"].update(
        action="squeezing under the garden gate",
        emotion="naughty",
        pose="crawling",
        outfit="blue jacket",
        direction="low angle, garden visible beyond",
        source="manual",
    )
    pan2.scripts["mrs-rabbit"].update(
        action="walking away, back turned",
        emotion="unaware",
        pose="walking",
        outfit="apron",
        source="manual",
    )

    # === Chapter 2: The Garden ===
    ch2 = story.create_chapter(
        "The Garden", ["peter", "mcgregor"],
        synopsis="Peter explores Mr. McGregor's garden and gets caught.",
        default_location="Mr. McGregor's vegetable garden",
        default_time_of_day="midday",
    )
    ch2.negative_prompt = "indoor, burrow, cozy"

    p3 = ch2.pages[0]
    p3.setting = "vegetable garden, rows of lettuces, beans, radishes"
    p3.mood = "tense, dangerous"
    p3.action_context = "eating then fleeing"
    p3.time_of_day = "midday"
    p3.weather = "clear"
    p3.lighting = "harsh sunlight, sharp shadows"

    pan3 = p3.panels[0]
    pan3.narration = "First he ate some lettuces and some French beans; and then he ate some radishes."
    pan3.shot_type = "close-up"
    pan3.scripts["peter"].update(
        action="eating lettuces, cheeks stuffed",
        emotion="happy",
        pose="sitting among vegetables",
        outfit="blue jacket",
        direction="close-up, surrounded by vegetables",
        source="manual",
    )
    # McGregor hasn't appeared yet
    if "mcgregor" in pan3.scripts and len(pan3.scripts) > 1:
        del pan3.scripts["mcgregor"]

    # Panel 2: McGregor appears
    pan4 = p3.create_panel(character_ids=["peter", "mcgregor"])
    story.register_panel(pan4)
    for s in pan4.scripts.values():
        story.register_script(s)

    pan4.narration = "Round the end of a cucumber frame, whom should he meet but Mr. McGregor!"
    pan4.shot_type = "wide"
    pan4.scripts["peter"].update(
        action="frozen in shock, lettuce falling from paws",
        emotion="scared",
        pose="crouching, frozen",
        outfit="blue jacket",
        direction="wide-eyed terror, ears back",
        source="manual",
    )
    pan4.scripts["mcgregor"].update(
        dialogue="Stop thief!",
        action="jumping up from planting, waving a rake",
        emotion="angry",
        pose="standing, lunging forward",
        outfit="overalls, straw hat",
        direction="looming over Peter",
        source="manual",
    )

    # Page 4: The chase
    p4 = ch2.create_page()
    p4.setting = "garden paths, vegetable beds, tool shed, gooseberry nets"
    p4.mood = "frantic, desperate"
    p4.action_context = "chase"
    p4.time_of_day = "midday"
    p4.weather = "clear"
    p4.lighting = "harsh sunlight"

    pan5 = p4.panels[0]
    pan5.narration = "Peter was most dreadfully frightened; he rushed all over the garden."
    pan5.shot_type = "wide"
    pan5.scripts["peter"].update(
        action="running desperately, tangled in gooseberry net",
        emotion="scared",
        pose="running, stumbling",
        outfit="blue jacket (torn)",
        direction="dynamic action, motion blur",
        source="manual",
    )
    pan5.scripts["mcgregor"].update(
        action="chasing with a sieve",
        emotion="angry",
        pose="running, arm outstretched",
        outfit="overalls",
        direction="background, pursuing",
        source="manual",
    )

    # Panel: Peter escaping jacket
    pan6 = p4.create_panel(character_ids=["peter"])
    story.register_panel(pan6)
    for s in pan6.scripts.values():
        story.register_script(s)

    pan6.narration = "Peter wriggled out just in time, leaving his jacket behind."
    pan6.shot_type = "medium"
    pan6.scripts["peter"].update(
        action="pulling free from jacket, running naked",
        emotion="scared",
        pose="twisting away",
        outfit="naked",
        direction="blue jacket left behind on the net",
        source="manual",
    )

    # === Chapter 3: Home Safe ===
    ch3 = story.create_chapter(
        "Home Safe", ["peter", "mrs-rabbit"],
        synopsis="Peter escapes and returns home, sick and sorry.",
        default_location="rabbit burrow",
        default_time_of_day="evening",
    )

    p5 = ch3.pages[0]
    p5.setting = "garden gate, then the path home through the wood"
    p5.mood = "relief, exhaustion"
    p5.action_context = "escape"
    p5.time_of_day = "afternoon"
    p5.weather = "clear"
    p5.lighting = "golden afternoon light"

    pan7 = p5.panels[0]
    pan7.narration = "He slipped underneath the gate, and was safe at last in the wood outside."
    pan7.shot_type = "medium"
    pan7.scripts["peter"].update(
        action="sliding under the gate, panting",
        emotion="relief",
        pose="crawling, exhausted",
        outfit="naked",
        direction="Peter squeezing under gate to freedom",
        source="manual",
    )
    if "mrs-rabbit" in pan7.scripts and len(pan7.scripts) > 1:
        del pan7.scripts["mrs-rabbit"]

    # Final page: bedtime
    p6 = ch3.create_page()
    p6.setting = "cozy rabbit burrow, Peter's bed"
    p6.mood = "quiet, lesson learned"
    p6.action_context = "bedtime"
    p6.time_of_day = "evening"
    p6.weather = "clear"
    p6.lighting = "warm candlelight"

    pan8 = p6.panels[0]
    pan8.narration = "Peter was not very well during the evening. His mother put him to bed."
    pan8.shot_type = "close-up"
    pan8.scripts["peter"].update(
        action="lying in bed, holding a cup of tea",
        emotion="sick",
        pose="lying in bed, propped on pillow",
        outfit="naked",
        direction="Peter in bed, droopy ears",
        source="manual",
    )
    pan8.scripts["mrs-rabbit"].update(
        dialogue="One table-spoonful to be taken at bed-time.",
        action="giving Peter camomile tea",
        emotion="concerned",
        pose="sitting beside bed, leaning forward",
        outfit="apron",
        direction="tender motherly moment",
        source="manual",
    )

    # Re-register all cascaded objects
    for ch in story.chapters.values():
        story._register_cascade(ch)

    # Set relationships
    peter.profile.set_relationship(
        "mrs-rabbit", "his mother — loves her but disobeys her warnings"
    )
    peter.profile.set_relationship(
        "mcgregor", "his nemesis — terrified of him but can't resist the garden"
    )
    mrs_rabbit.profile.set_relationship(
        "peter", "her naughty youngest son — worries about him constantly"
    )
    mcgregor.profile.set_relationship(
        "peter", "a pest who raids his garden — wants to catch him"
    )

    # Validate
    errors = story.validate()
    if errors:
        print(f"Validation errors: {errors}")
        return

    # Save
    output_path = save_story(story, content_store, "data/stories/peter_rabbit")
    print(f"Saved to: {output_path}")
    print(f"Characters: {len(story.characters)}")
    print(f"Chapters: {len(story.chapters)}")
    total_pages = sum(
        len(ch.pages) for ch in story.chapters.values() if not ch.is_solo
    )
    total_panels = sum(
        len(p.panels) for ch in story.chapters.values()
        if not ch.is_solo for p in ch.pages
    )
    total_scripts = sum(
        len(pan.scripts) for ch in story.chapters.values()
        if not ch.is_solo for p in ch.pages for pan in p.panels
    )
    print(f"Pages: {total_pages}")
    print(f"Panels: {total_panels}")
    print(f"Scripts: {total_scripts}")
    print(f"Art style: {story.art_style}")
    print(f"Story negative: {story.negative_prompt}")


if __name__ == "__main__":
    create_peter_rabbit()
