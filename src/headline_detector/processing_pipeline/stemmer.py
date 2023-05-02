"""
This is a wraper class to stemming without effecting changed text such as
emoji or 'HTTPURL'
"""
import re
from typing import Optional
from . import TextProcessingPipeline
from NDETCStemmer import NDETCStemmer

class NDETCStemmerWraper:
    
    def __init__(self, stemmer: 'Optional[NDETCStemmer]' = None) -> None:
        if stemmer is not None:
            self._stemmer = stemmer
        else:
            self._stemmer = NDETCStemmer()
        self._text_preprocessor = TextProcessingPipeline([
            self._change_emoji,
            self._change_httpurl,
            self._change_user
        ])
        self._text_postprocessor = TextProcessingPipeline([
            self._dechange_stem_emoji,
            self._dechange_stem_httpurl,
            self._dechange_stem_user
        ])
        
    def stem(self, text:str) -> str:
        pre_processed_text = self._text_preprocessor.process_text(text)
        stemmed_text = self._stemmer.stem(pre_processed_text)
        post_processed_text = self._text_postprocessor.process_text(stemmed_text)
        return post_processed_text
        
    def _change_httpurl(self, text: str) ->str:
        return re.sub("HTTPURL", "HTTPURLstem", text, count=0)

    def _change_user(self, text: str) ->str:
        return re.sub("@USER", "userstem", text, count=0)

    def _change_emoji(self, text: str) ->str:
        final_data = text
        emo_text = re.findall(r"(:(\w+?):)", text)
        for emo in emo_text:
            full_emoji = emo[0]
            emoji_description = emo[1]
            replacement = "emotji"+emoji_description+"emotji"
            final_data = re.sub(full_emoji, replacement, final_data, count=0)
        return final_data
    
    def _dechange_stem_httpurl(self, text:str) -> str:
        return re.sub("httpurlstem", "HTTPURL", text, count=0)

    def _dechange_stem_user(self, text:str) -> str:
        return re.sub("userstem", "@USER", text, count=0)

    def _dechange_stem_emoji(self, text:str) -> str:
        final_data = text
        emo_text = re.findall(r"(emotji(\w+?)emotji)", text)
        for emo in emo_text:
            full_emoji = emo[0]
            emoji_description = emo[1]
            original = ":"+emoji_description+":"
            final_data = re.sub(full_emoji, original, final_data, count=0)
        return final_data

# Example on how to use it        
if __name__ == "__main__":
    dummy_text = ":test_1::test2: HTTPURL miliarder @USER rusia oleg tinkov pada senin (31/10/2022), mengaku telah melepaskan kewarganegaraan rusianya karena konflik di ukraina. HTTPURL @USER @USER :sad_but_relieved_face: :test_1: :test2:"
    
    # Instantiate NDETCStemmerWraper
    stemmer = NDETCStemmerWraper()
    print(stemmer.stem(dummy_text))