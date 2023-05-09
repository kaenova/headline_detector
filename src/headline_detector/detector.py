import torch
import typing
import warnings
import huggingface_hub as hf_hub
import torch.nn.functional as F

from tqdm import tqdm
from .processing_pipeline import TextProcessingPipeline

from . import model
from . import utils
from . import hyper_params as h_params
from . import model_checkpoint as m_checkpoint
from . import preprocessing_scenario as prep_scenario


class FasttextDetector:
    def __init__(
        self,
        model: "model.FastTextClassifier",
        preprocessor: "typing.Optional[TextProcessingPipeline]" = None,
    ) -> None:
        model.eval()
        self.active_model = model
        self.preprocessor = preprocessor

    def predict_text(self, texts: "typing.Union[str, list[str]]") -> "list[int]":
        if type(texts) == str:
            texts = [texts]
        texts = utils.ensure_list_string(texts)
        prediction = []
        with torch.no_grad():
            for text in tqdm(texts):
                if self.preprocessor is not None:
                    text = self.preprocessor.process_text(text)
                logits = self.active_model.forward([text])
                prediction.append(logits)
        prediction = torch.cat(prediction, 0)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, 1)
        prediction = prediction.cpu().numpy()
        prediction = prediction.tolist()
        return prediction

    @staticmethod
    def load_from_scenario(scenario_num: "int") -> "FasttextDetector":
        # Load model
        scenario_num = int(scenario_num)
        if scenario_num not in m_checkpoint.fasttext_checkpoints.keys():
            raise ValueError("Not a valid scenario number")
        model_path = hf_hub.hf_hub_download(
            repo_id=m_checkpoint.repo_id,
            filename=m_checkpoint.fasttext_checkpoints[scenario_num],
            cache_dir=".headline_detector_model",
        )
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        hyper_params = h_params.hyper_params_fasttext
        loaded_model = model.FastTextClassifier(
            hyper_params["seq_length"],
            hyper_params["out_feature"],
        )
        loaded_model.load_state_dict(state_dict=state_dict)
        loaded_model.eval()

        # Load preprocessor
        if scenario_num not in prep_scenario.scenario_processor.keys():
            warnings.warn(
                "preprocessing is not defined, will not preprocess the text when predicting"
            )
        preprocessor = None
        if scenario_num in prep_scenario.scenario_processor.keys():
            preprocessor = prep_scenario.scenario_processor[scenario_num]

        return FasttextDetector(loaded_model, preprocessor)


class CNNDetector:
    def __init__(
        self,
        model: "model.CNNClassifier",
        preprocessor: "typing.Optional[TextProcessingPipeline]" = None,
    ) -> None:
        model.eval()
        self.active_model = model
        self.preprocessor = preprocessor

    def predict_text(self, texts: "typing.Union[str, list[str]]") -> "list[int]":
        if type(texts) == str:
            texts = [texts]
        texts = utils.ensure_list_string(texts)
        prediction = []
        with torch.no_grad():
            for text in tqdm(texts):
                if self.preprocessor is not None:
                    text = self.preprocessor.process_text(text)
                logits = self.active_model.forward([text])
                prediction.append(logits)
        prediction = torch.cat(prediction, 0)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, 1)
        prediction = prediction.cpu().numpy()
        prediction = prediction.tolist()
        return prediction

    @staticmethod
    def load_from_scenario(scenario_num: "int") -> "CNNDetector":
        scenario_num = int(scenario_num)
        if scenario_num not in m_checkpoint.cnn_checkpoints.keys():
            raise ValueError("Not a valid scenario number")
        model_path = hf_hub.hf_hub_download(
            repo_id=m_checkpoint.repo_id,
            filename=m_checkpoint.cnn_checkpoints[scenario_num],
            cache_dir=".headline_detector_model",
        )
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        hyper_params = h_params.hyper_params_cnn
        loaded_model = model.CNNClassifier(
            hyper_params["seq_length"],
            hyper_params["out_feature"],
            conv_num_filters=hyper_params["conv_num_filters"],
            conv_kernels=hyper_params["conv_kernels"],
        )
        loaded_model.load_state_dict(state_dict=state_dict)
        loaded_model.eval()

        # Load preprocessor
        if scenario_num not in prep_scenario.scenario_processor.keys():
            warnings.warn(
                "preprocessing is not defined, will not preprocess the text when predicting"
            )
        preprocessor = None
        if scenario_num in prep_scenario.scenario_processor.keys():
            preprocessor = prep_scenario.scenario_processor[scenario_num]

        return CNNDetector(loaded_model, preprocessor)


class IndoBERTweetDetector:
    def __init__(
        self,
        model: "model.FastTextClassifier",
        preprocessor: "typing.Optional[TextProcessingPipeline]" = None,
    ) -> None:
        model.eval()
        self.active_model = model
        self.preprocessor = preprocessor

    def predict_text(self, texts: "typing.Union[str, list[str]]") -> "list[int]":
        if type(texts) == str:
            texts = [texts]
        texts = utils.ensure_list_string(texts)
        prediction = []
        with torch.no_grad():
            for text in tqdm(texts):
                if self.preprocessor is not None:
                    text = self.preprocessor.process_text(text)
                logits = self.active_model.forward([text])
                prediction.append(logits)
        prediction = torch.cat(prediction, 0)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, 1)
        prediction = prediction.cpu().numpy()
        prediction = prediction.tolist()
        return prediction

    @staticmethod
    def load_from_scenario(scenario_num: "int") -> "IndoBERTweetDetector":
        scenario_num = int(scenario_num)
        if scenario_num not in m_checkpoint.indobertweet_checkpoints.keys():
            raise ValueError("Not a valid scenario number")
        model_path = hf_hub.hf_hub_download(
            repo_id=m_checkpoint.repo_id,
            filename=m_checkpoint.indobertweet_checkpoints[scenario_num],
            cache_dir=".headline_detector_model",
        )
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        hyper_params = h_params.hyper_params_indobertweet
        loaded_model = model.BERTClassifier(
            hyper_params["model_name"],
            hyper_params["seq_length"],
            hyper_params["out_feature"],
        )
        loaded_model.load_state_dict(state_dict=state_dict)
        loaded_model.eval()

        # Load preprocessor
        if scenario_num not in prep_scenario.scenario_processor.keys():
            warnings.warn(
                "preprocessing is not defined, will not preprocess the text when predicting"
            )
        preprocessor = None
        if scenario_num in prep_scenario.scenario_processor.keys():
            preprocessor = prep_scenario.scenario_processor[scenario_num]

        return IndoBERTweetDetector(loaded_model, preprocessor)
