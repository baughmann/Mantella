import pytest
from src.character_manager import Character
from src.characters_manager import Characters
from src.llm.output.change_character_parser import change_character_parser
from src.llm.output.output_parser import sentence_generation_settings


@pytest.fixture
def characters(example_skyrim_player_character: Character, example_skyrim_npc_character: Character) -> Characters:
    chars = Characters()
    chars.add_or_update_character(example_skyrim_player_character)
    chars.add_or_update_character(example_skyrim_npc_character)
    return chars


def test_switch_to_known_npc(characters: Characters, example_skyrim_npc_character: Character):
    settings = sentence_generation_settings(example_skyrim_npc_character)
    result, rest = change_character_parser(characters).cut_sentence(f"{example_skyrim_npc_character.name}: Hello there!", settings)
    assert result is None
    assert rest == " Hello there!"
    assert settings.current_speaker is example_skyrim_npc_character
    assert not settings.stop_generation


def test_player_prefix_stops_generation(characters: Characters, example_skyrim_npc_character: Character):
    settings = sentence_generation_settings(example_skyrim_npc_character)
    result, rest = change_character_parser(characters).cut_sentence("player: I should not say this", settings)
    assert result is None
    assert rest == ""
    assert settings.stop_generation


def test_unnamed_player_does_not_swallow_npc_lines(characters: Characters, example_skyrim_player_character: Character, example_skyrim_npc_character: Character):
    """A player with an empty name (e.g. a `coc` new game in Fallout 4) must not match every 'Name:' prefix."""
    example_skyrim_player_character.name = ''
    settings = sentence_generation_settings(example_skyrim_npc_character)
    result, rest = change_character_parser(characters).cut_sentence(f"{example_skyrim_npc_character.name}: Safe travels", settings)
    assert not settings.stop_generation
    assert result is None
    assert rest == " Safe travels"
    assert settings.current_speaker is example_skyrim_npc_character
