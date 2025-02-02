from loguru import logger
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage

from macrec.llms.basellm import BaseLLM

import base64
import httpx

class AnyOpenAILLM(BaseLLM):
    def __init__(self, model_name: str = 'gpt-3.5-turbo', json_mode: bool = False, img_processing: bool = False, *args, **kwargs):
        """Initialize the OpenAI LLM.
        
        Args:
            `model_name` (`str`, optional): The name of the OpenAI model. Defaults to `gpt-3.5-turbo`.
            `json_mode` (`bool`, optional): Whether to use the JSON mode of the OpenAI API. Defaults to `False`.
        """
        self.model_name = model_name
        self.json_mode = json_mode
        self.img_processing = img_processing
        # if json_mode and self.model_name not in ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']:
        #     raise ValueError("json_mode is only available for gpt-3.5-turbo-1106 and gpt-4-1106-preview")
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        self.max_context_length: int = 16384 if '16k' in model_name else 32768 if '32k' in model_name else 4096
        if model_name.split('-')[0] == 'text' or model_name == 'gpt-3.5-turbo-instruct':
            self.model = OpenAI(model_name=model_name, *args, **kwargs)
            self.model_type = 'completion'
        else:
            if json_mode:
                logger.info("Using JSON mode of OpenAI API.")
                if 'model_kwargs' in kwargs:
                    kwargs['model_kwargs']['response_format'] = {
                        "type": "json_object"
                    }
                else:
                    kwargs['model_kwargs'] = {
                        "response_format": {
                            "type": "json_object"
                        }
                    }
            self.model = ChatOpenAI(model_name=model_name, *args, **kwargs)
            if img_processing:
                self.model_type = 'img'
            else:
                self.model_type = 'chat'
    
    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the OpenAI LLM.
        
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenAI LLM output.
        """
        if self.model_type == 'completion':
            return self.model.invoke(prompt).content.replace('\n', ' ').strip()
        elif self.model_type == 'img':
            # This is the first implementation of image_processing, which is very detailed and not so clean and reusable
            # TODO: I may change the code structure when I deeply understand the codebase
            image_url = prompt
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": 'describe the image'},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            return self.model.invoke([message]).content
        else:
            return self.model.invoke(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content.replace('\n', ' ').strip()