import logging
import re
from src.llm.output.output_parser import output_parser, sentence_generation_settings
from src.llm.sentence_content import SentenceContent

class clean_sentence_parser(output_parser):
    """Class to track narrations in the current output of the LLM."""
    def __init__(self, strip_markdown: bool = False) -> None:
        """strip_markdown: remove Markdown emphasis instead of treating asterisks as narration.
        Only valid when '*' is not configured as a narration indicator."""
        super().__init__()
        self.__strip_markdown = strip_markdown

    def cut_sentence(self, output: str, current_settings: sentence_generation_settings) -> tuple[SentenceContent | None, str]:
        return None, self.clean_sentence(output)
        
    def clean_sentence(self, sentence: str) -> str:
        def remove_as_a(sentence: str) -> str:
            """Remove 'As an XYZ,' from beginning of sentence"""
            if sentence.startswith('As a'):
                if ', ' in sentence:
                    logging.log(28, f"Removed '{sentence.split(', ')[0]} from response")
                    sentence = sentence.replace(sentence.split(', ')[0]+', ', '')
            return sentence
        
        if ('Well, well, well' in sentence):
            sentence = sentence.replace('Well, well, well', 'Well well well')

        sentence = remove_as_a(sentence)
        sentence = sentence.replace('\r\n', ' ')
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('[', '(')
        sentence = sentence.replace(']', ')')
        sentence = sentence.replace('{', '(')
        sentence = sentence.replace('}', ')')
        if self.__strip_markdown:
            # Models like to answer in Markdown. With only () / [] marking narration, emphasis such as
            # *really*, **bold**, _italic_, `code` or ~~struck~~ is spoken text: drop the markup, keep the words.
            # This runs on streamed chunks, so remove the characters rather than matching pairs.
            sentence = sentence.replace('*', '').replace('`', '').replace('~~', '')
            sentence = re.sub(r'(?<!\w)_+|_+(?!\w)', '', sentence)
        else:
            # local models sometimes get the idea in their head to use double asterisks **like this** in sentences instead of single
            # this converts double asterisks to single so that they can be filtered out appropriately
            sentence = sentence.replace('**','*')
        return sentence

    def modify_sentence_content(self, cut_content: SentenceContent, last_content: SentenceContent | None, settings: sentence_generation_settings) -> tuple[SentenceContent | None, SentenceContent | None]:
        return cut_content, last_content