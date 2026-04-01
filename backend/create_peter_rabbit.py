"""Create The Tale of Peter Rabbit as a demo .cvn story file.

Public domain text by Beatrix Potter (1902).
Characters, chapters, pages, panels, and scripts — all populated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import Story, Character, ContentStore
from backend.models.storage import save_story
from backend.models.profile import Profile, PhysicalTraits, Outfit


def create_peter_rabbit():
    story = Story("peter-rabbit", "The Tale of Peter Rabbit")
    story.synopsis = "A naughty young rabbit disobeys his mother and sneaks into Mr. McGregor's garden, narrowly escaping with his life."

    content_store = ContentStore("data/content")

    # === Characters ===

    peter = Character("peter", "Peter Rabbit",
        description="A naughty, curious young rabbit who can't resist adventure. Impulsive and mischievous but ultimately learns his lesson.",
        personality_prompt="curious, naughty, impulsive, brave but easily frightened",
        appearance_prompt="young brown rabbit, blue jacket with brass buttons, upright ears, bright eyes",
    )
    peter.profile.biography = "The youngest and most mischievous of four rabbit siblings. His father was caught by Mr. McGregor and put in a pie — a warning Peter ignores."
    peter.profile.physical = PhysicalTraits(
        body="small young rabbit, lean and quick",
        face="bright curious eyes, twitchy nose, long upright ears",
        distinguishing_marks="wears a blue jacket with brass buttons (when he hasn't lost it)",
        hair_fur="soft brown fur, white cotton tail",
    )
    peter.profile.add_outfit("blue jacket", "blue jacket with brass buttons, quite new", is_default=True)
    peter.profile.add_outfit("naked", "no clothes — lost his jacket in McGregor's garden")
    peter.profile.tendencies = ["gets into trouble", "runs headfirst into danger", "cries when scared"]
    peter.profile.expressions = {
        "scared": "ears flat back, eyes wide, trembling",
        "naughty": "mischievous grin, ears perked forward",
        "sick": "droopy ears, half-closed eyes, hunched over",
        "crying": "big tears, ears drooping, sitting hunched",
    }
    story.add_character(peter)

    mrs_rabbit = Character("mrs-rabbit", "Mrs. Rabbit",
        description="Peter's mother. A sensible, caring rabbit who warns her children about Mr. McGregor's garden.",
        personality_prompt="caring, sensible, maternal, firm but loving",
        appearance_prompt="adult brown rabbit in an apron, warm motherly expression, carrying a basket",
    )
    mrs_rabbit.profile.physical = PhysicalTraits(
        body="plump motherly rabbit",
        face="kind eyes, gentle expression",
        hair_fur="brown fur, neat and well-groomed",
    )
    mrs_rabbit.profile.add_outfit("apron", "neat apron, carries a basket and umbrella", is_default=True)
    story.add_character(mrs_rabbit)

    mcgregor = Character("mcgregor", "Mr. McGregor",
        description="A grumpy old gardener who chases rabbits out of his vegetable garden. Peter's nemesis.",
        personality_prompt="grumpy, determined, territorial about his garden",
        appearance_prompt="old human gardener, overalls, straw hat, carrying a rake, angry expression",
    )
    mcgregor.profile.physical = PhysicalTraits(
        body="stocky old man, strong from gardening",
        face="weathered face, bushy eyebrows, scowling",
    )
    mcgregor.profile.add_outfit("gardening", "dirty overalls, boots, straw hat", is_default=True)
    mcgregor.profile.tendencies = ["shakes his fist", "waves garden tools threateningly", "stomps around"]
    story.add_character(mcgregor)

    # === Chapter 1: The Warning ===
    ch1 = story.create_chapter("The Warning", ["peter", "mrs-rabbit"],
        synopsis="Mrs. Rabbit warns her children not to go into Mr. McGregor's garden.")

    # Page 1: The family
    p1 = ch1.pages[0]
    p1.setting = "cozy rabbit burrow under a big fir tree, warm morning light"
    p1.mood = "warm, domestic"
    p1.action_context = "morning routine"

    pan1 = p1.panels[0]
    pan1.narration = "Once upon a time there were four little Rabbits."
    pan1.scripts["peter"].update(action="sitting with siblings", emotion="restless", source="manual")
    pan1.scripts["mrs-rabbit"].update(
        dialogue="Now my dears, you may go into the fields, but don't go into Mr. McGregor's garden.",
        action="standing at the burrow entrance with basket and umbrella",
        emotion="serious",
        direction="medium shot, warm lighting",
        source="manual",
    )

    # Page 2: Peter sneaks off
    p2 = ch1.create_page()
    p2.setting = "garden gate, lush vegetable garden beyond the fence"
    p2.mood = "mischievous, exciting"
    p2.action_context = "sneaking"

    pan2 = p2.panels[0]
    pan2.narration = "But Peter, who was very naughty, ran straight away to Mr. McGregor's garden."
    pan2.scripts["peter"].update(
        action="squeezing under the garden gate",
        emotion="naughty",
        direction="low angle showing Peter squeezing under the gate, garden visible beyond",
        source="manual",
    )
    pan2.scripts["mrs-rabbit"].update(
        action="walking away with basket, back turned",
        emotion="unaware",
        source="manual",
    )

    # === Chapter 2: The Garden ===
    ch2 = story.create_chapter("The Garden", ["peter", "mcgregor"],
        synopsis="Peter explores Mr. McGregor's garden and gets caught.")

    p3 = ch2.pages[0]
    p3.setting = "Mr. McGregor's vegetable garden, rows of lettuces, beans, radishes, cucumber frames"
    p3.mood = "tense, dangerous"
    p3.action_context = "eating then fleeing"

    pan3 = p3.panels[0]
    pan3.narration = "First he ate some lettuces and some French beans; and then he ate some radishes."
    pan3.scripts["peter"].update(
        action="eating lettuces, cheeks stuffed, sitting among the vegetables",
        emotion="happy",
        direction="close-up, surrounded by vegetables",
        source="manual",
    )
    # McGregor hasn't appeared yet — remove him from this panel
    if "mcgregor" in pan3.scripts and len(pan3.scripts) > 1:
        del pan3.scripts["mcgregor"]

    # Add a second panel: McGregor appears
    pan4 = p3.create_panel(character_ids=["peter", "mcgregor"])
    story.register_panel(pan4)
    for s in pan4.scripts.values():
        story.register_script(s)

    pan4.narration = "Round the end of a cucumber frame, whom should he meet but Mr. McGregor!"
    pan4.scripts["peter"].update(
        action="frozen in shock, lettuce falling from paws",
        emotion="scared",
        direction="wide-eyed terror, ears back",
        source="manual",
    )
    pan4.scripts["mcgregor"].update(
        dialogue="Stop thief!",
        action="jumping up from planting cabbages, waving a rake",
        emotion="angry",
        direction="looming over Peter, threatening",
        source="manual",
    )

    # Page 4: The chase
    p4 = ch2.create_page()
    p4.setting = "garden paths between vegetable beds, tool shed, gooseberry nets"
    p4.mood = "frantic, desperate"
    p4.action_context = "chase"

    pan5 = p4.panels[0]
    pan5.narration = "Peter was most dreadfully frightened; he rushed all over the garden."
    pan5.scripts["peter"].update(
        action="running desperately, losing shoes, tangled in gooseberry net",
        emotion="scared",
        direction="dynamic action shot, motion blur",
        source="manual",
    )
    pan5.scripts["mcgregor"].update(
        action="chasing with a sieve, trying to catch Peter",
        emotion="angry",
        direction="background, pursuing",
        source="manual",
    )

    # Add panel: Peter escaping
    pan6 = p4.create_panel(character_ids=["peter"])
    story.register_panel(pan6)
    for s in pan6.scripts.values():
        story.register_script(s)

    pan6.narration = "Peter wriggled out just in time, leaving his jacket behind him."
    pan6.scripts["peter"].update(
        action="squeezing out of the jacket, running naked",
        emotion="scared",
        direction="Peter pulling free, blue jacket left behind on the net",
        source="manual",
    )

    # === Chapter 3: Home Safe ===
    ch3 = story.create_chapter("Home Safe", ["peter", "mrs-rabbit"],
        synopsis="Peter escapes and returns home, sick and sorry.")

    p5 = ch3.pages[0]
    p5.setting = "garden gate, then the path home through the wood"
    p5.mood = "relief, exhaustion"
    p5.action_context = "escape"

    pan7 = p5.panels[0]
    pan7.narration = "He slipped underneath the gate, and was safe at last in the wood outside the garden."
    pan7.scripts["peter"].update(
        action="sliding under the gate, panting, no jacket, no shoes",
        emotion="relief",
        direction="Peter squeezing under the gate to freedom",
        source="manual",
    )
    # Mrs. Rabbit not in this scene
    if "mrs-rabbit" in pan7.scripts and len(pan7.scripts) > 1:
        del pan7.scripts["mrs-rabbit"]

    # Final page: bedtime
    p6 = ch3.create_page()
    p6.setting = "cozy rabbit burrow, warm candlelight, Peter's bed"
    p6.mood = "quiet, lesson learned"
    p6.action_context = "bedtime"

    pan8 = p6.panels[0]
    pan8.narration = "Peter was not very well during the evening. His mother put him to bed and made some camomile tea."
    pan8.scripts["peter"].update(
        dialogue="",
        action="lying in bed, looking miserable, holding a cup of tea",
        emotion="sick",
        direction="Peter in bed, droopy ears, Mrs. Rabbit giving him tea",
        source="manual",
    )
    pan8.scripts["mrs-rabbit"].update(
        dialogue="One table-spoonful to be taken at bed-time.",
        action="giving Peter a spoonful of camomile tea",
        emotion="concerned",
        direction="tender motherly moment",
        source="manual",
    )

    # Re-register all cascaded objects
    for ch in story.chapters.values():
        story._register_cascade(ch)

    # Set relationships
    peter.profile.set_relationship("mrs-rabbit", "his mother — loves her but disobeys her warnings")
    peter.profile.set_relationship("mcgregor", "his nemesis — terrified of him but can't resist the garden")
    mrs_rabbit.profile.set_relationship("peter", "her naughty youngest son — worries about him constantly")
    mcgregor.profile.set_relationship("peter", "a pest who raids his garden — wants to catch him")

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
    total_pages = sum(len(ch.pages) for ch in story.chapters.values())
    total_panels = sum(len(p.panels) for ch in story.chapters.values() for p in ch.pages)
    total_scripts = sum(len(pan.scripts) for ch in story.chapters.values() for p in ch.pages for pan in p.panels)
    print(f"Pages: {total_pages}")
    print(f"Panels: {total_panels}")
    print(f"Scripts: {total_scripts}")


if __name__ == "__main__":
    create_peter_rabbit()
