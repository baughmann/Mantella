"""Narration = () / [] only; everything else, including Markdown emphasis and quotes, is spoken.

Inputs are real gemma-4-31b-it replies from in-game tests that previously lost their dialogue.
The helper mirrors the parse loop in ChatManager (output_manager.py) with narration cut.
"""
import pytest
from src.character_manager import Character
from src.characters_manager import Characters
from src.llm.output.change_character_parser import change_character_parser
from src.llm.output.narration_parser import narration_parser
from src.llm.output.output_parser import output_parser, sentence_generation_settings
from src.llm.output.sentence_accumulator import sentence_accumulator
from src.llm.output.sentence_end_parser import END_OF_OUTPUT, sentence_end_parser
from src.llm.sentence_content import SentenceTypeEnum

NARRATION_START = ["(", "["]
NARRATION_END = [")", "]"]


def spoken_lines(text: str, characters: Characters, speaker: Character) -> list[tuple[str, str]]:
    """Parse `text` streamed in 1-, 3- and 50-character chunks; the result must not depend on chunking."""
    results = [_spoken_lines(text, characters, speaker, chunk) for chunk in (1, 3, 50)]
    assert results[0] == results[1] == results[2], f"chunking changed the output: {results}"
    return results[0]


def _spoken_lines(text: str, characters: Characters, speaker: Character, chunk: int) -> list[tuple[str, str]]:
    chain: list[output_parser] = [
        change_character_parser(characters),
        narration_parser(NARRATION_START, NARRATION_END, [], []),
        sentence_end_parser(),
    ]
    accumulator = sentence_accumulator(list({i for p in chain for i in p.get_cut_indicators()}))
    settings = sentence_generation_settings(speaker)
    pending = None
    out: list[tuple[str, str]] = []
    # stream it token-ish, like the LLM does, then the end-of-stream flush ChatManager adds
    chunks = [text[start:start + chunk] for start in range(0, len(text), chunk)] + [END_OF_OUTPUT]
    for piece in chunks:
        accumulator.accumulate(piece)
        while accumulator.has_next_sentence():
            current = accumulator.get_next_sentence()
            parsed = None
            for parser in chain:
                if not parsed:
                    parsed, current = parser.cut_sentence(current, settings)
                if parsed:
                    parsed, pending = parser.modify_sentence_content(parsed, pending, settings)
                if settings.stop_generation:
                    break
            accumulator.refuse(current)
            if parsed and parsed.sentence_type != SentenceTypeEnum.NARRATION and parsed.text.strip():
                out.append((parsed.speaker.name, parsed.text.strip()))
            if settings.stop_generation:
                return out
    return out


@pytest.fixture
def cast(example_skyrim_player_character: Character, example_skyrim_npc_character: Character) -> tuple[Characters, Character]:
    example_skyrim_npc_character.name = "Cricket"
    chars = Characters()
    chars.add_or_update_character(example_skyrim_player_character)
    chars.add_or_update_character(example_skyrim_npc_character)
    return chars, example_skyrim_npc_character


def test_quoted_word_is_spoken(cast):
    chars, npc = cast
    lines = spoken_lines('That "Mantella" thing? Not a chance. I\'ve never heard of such a thing.', chars, npc)
    assert [t for _, t in lines] == ['That "Mantella" thing?', 'Not a chance.', "I've never heard of such a thing."]


def test_parenthesised_reply_is_narration(cast):
    chars, npc = cast
    assert spoken_lines("(The lead trader steps forward, adjusting his pack with a huff.)", chars, npc) == []


def test_markdown_emphasis_is_spoken(cast):
    chars, npc = cast
    lines = spoken_lines("**Really** good stuff, *friend*. You won't find _better_ prices.", chars, npc)
    assert [t for _, t in lines] == ["Really good stuff, friend.", "You won't find better prices."]


def test_star_bracket_stage_direction_is_cut_but_dialogue_kept(cast):
    chars, npc = cast
    text = ("Cricket: *[Her voice is thin and wavering, trailing off as she scratches her arm]* Oh... hey there. "
            "Not too bad. [She lets out a shaky sigh.] Anyway, you looking for something?")
    lines = spoken_lines(text, chars, npc)
    assert all(speaker == "Cricket" for speaker, _ in lines)
    spoken = " ".join(t for _, t in lines)
    assert "hey there" in spoken and "Not too bad." in spoken and "you looking for something?" in spoken
    assert "*" not in spoken and "scratches" not in spoken and "sigh" not in spoken


def test_ellipsis_does_not_split_a_sentence(cast):
    """Real reply that was voiced as three fragments: 'To get to the...' / 'the other side...' / 'of the radiation zone.'"""
    chars, npc = cast
    lines = spoken_lines("Uh... why did the ghoul cross the road? To get to the... the other side... of the radiation zone.", chars, npc)
    assert [t for _, t in lines] == ["Uh... why did the ghoul cross the road?", "To get to the... the other side... of the radiation zone."]


def test_reply_ending_in_ellipsis_is_still_spoken(cast):
    chars, npc = cast
    lines = spoken_lines("A joke? Now that's... a new one. If you know who to sell to...", chars, npc)
    assert [t for _, t in lines] == ["A joke?", "Now that's... a new one.", "If you know who to sell to..."]


def test_tail_without_punctuation_is_flushed(cast):
    chars, npc = cast
    lines = spoken_lines("Safe travels. Watch the road", chars, npc)
    assert [t for _, t in lines] == ["Safe travels.", "Watch the road"]
