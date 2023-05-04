from .processing_pipeline import TextProcessingPipeline, NDETCStemmerWraper
from .processing_pipeline import processing_func as pf

from NDETCStemmer import NDETCStemmer

original_stemmer = NDETCStemmer()
stemmer = NDETCStemmerWraper(original_stemmer)

scenario_processor = {
    1: TextProcessingPipeline(
        [pf.lowercasing, pf.change_user, pf.change_emoji, pf.change_web_url]
    ),
    2: TextProcessingPipeline(
        [
            pf.lowercasing,
            pf.change_user,
            pf.change_emoji,
            pf.change_web_url,
            stemmer.stem,
        ]
    ),
    3: TextProcessingPipeline(
        [
            pf.lowercasing,
            pf.remove_username,
            pf.remove_url,
            pf.change_emoji,
        ]
    ),
    4: TextProcessingPipeline(
        [
            pf.lowercasing,
            pf.remove_username,
            pf.remove_url,
            pf.change_emoji,
            stemmer.stem,
        ]
    ),
    5: TextProcessingPipeline(
        [
            pf.lowercasing,
            pf.remove_username,
            pf.remove_url,
            pf.remove_emoji,
        ]
    ),
    6: TextProcessingPipeline(
        [
            pf.lowercasing,
            pf.remove_username,
            pf.remove_url,
            pf.remove_emoji,
            stemmer.stem,
        ]
    ),
}