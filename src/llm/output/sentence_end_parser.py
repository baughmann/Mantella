import re
import unicodedata
from src.llm.output.output_parser import output_parser, sentence_generation_settings
from src.llm.sentence_content import SentenceContent

# Appended once when the LLM stream is complete, so text after the last sentence end
# (e.g. a reply that trails off with "...") is still cut into a final sentence.
END_OF_OUTPUT = "\x03"


class sentence_end_parser(output_parser):
    """Class to cut the LLM output at the end of a sentence."""
    def __init__(self, end_of_sentence_chars: list[str] = ['.', '?', '!', ';', '。', '？', '！', '；', '：']) -> None:
        super().__init__()
        self.__end_of_sentence_chars = [unicodedata.normalize('NFKC', char) for char in end_of_sentence_chars]
        self.__end_run_reg = re.compile("[{chars}]+".format(chars = "\\" + "\\".join(self.__end_of_sentence_chars + [END_OF_OUTPUT])))

    def cut_sentence(self, output: str, current_settings: sentence_generation_settings) -> tuple[SentenceContent | None, str]:
        for match in self.__end_run_reg.finditer(output):
            run = match.group()
            # An ellipsis ("..", "...") is a pause, not the end of a sentence: splitting there turned
            # "To get to the... the other side..." into separately voiced fragments.
            if END_OF_OUTPUT not in run and len(run) > 1 and set(run) == {'.'}:
                continue
            # A '.' at the very end of what has streamed so far may be the first dot of an ellipsis:
            # wait for the next character (sentence_accumulator hands it over as soon as it arrives).
            if END_OF_OUTPUT not in run and run == '.' and match.end() == len(output):
                return None, output
            matched_text = output[:match.end()].replace(END_OF_OUTPUT, '')
            rest = output[match.end():]
            if not matched_text.strip():
                return None, rest
            return SentenceContent(current_settings.current_speaker, matched_text, current_settings.sentence_type, False), rest
        return None, output

    def modify_sentence_content(self, cut_content: SentenceContent, last_content: SentenceContent | None, settings: sentence_generation_settings) -> tuple[SentenceContent | None, SentenceContent | None]:
        return cut_content, last_content

    def get_cut_indicators(self) -> list[str]:
        return self.__end_of_sentence_chars + [END_OF_OUTPUT]
