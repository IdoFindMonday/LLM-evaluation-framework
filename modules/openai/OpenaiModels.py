import openai


class OpenaiCompletionBase:
    def __init__(self, model_name="text-davinci-003",
                 api_version="2023-05-15", temperature=0.0, max_tokens=200, prompt_template=""):
        self.model_name = model_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self.top_p = 1
        self.n = 1

    def _get_response(self, **input_args):
        prompt = self.prompt_template.format(**input_args)
        response = openai.Completion.create(
            engine=self.model_name,
            api_version=self.api_version,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n
        )

        return response

    def call_llm(self, input_text):
        input_args = {"input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_args = {"input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text, response

    def get_config(self):
        return self.__dict__


class OpenaiCompletionWordLimit(OpenaiCompletionBase):
    def __init__(self, model_name="text-davinci-003",
                 api_version="2023-05-15", temperature=0.0, max_tokens=200, prompt_template="", max_word_ratio=0.6):
        super().__init__(model_name, api_version, temperature, max_tokens, prompt_template)
        self.max_word_ratio = max_word_ratio

    def call_llm(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        input_args = {"max_words": max_words, "input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        input_args = {"max_words": max_words, "input_text": input_text}
        response = self._get_response(**input_args)
        output_text = response["choices"][0]["text"].strip()
        return output_text, response


class OpenaiChatCompletionBase:
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template=""):
        self.model_name = model_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.top_p = 1
        self.n = 1

    def _get_response(self, messages):
        response = openai.ChatCompletion.create(
            engine=self.model_name,
            api_version=self.api_version,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n
        )

        return response

    def call_llm(self, input_text):
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": self.system_prompt_template},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})

        messages = [
            {"role": "system", "content": self.system_prompt_template},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response

    def get_config(self):
        return self.__dict__


class OpenaiChatCompletionWordLimit(OpenaiChatCompletionBase):
    def __init__(self, model_name="gpt-3-5-turbo", api_version="2023-05-15", temperature=0.0, max_tokens=200,
                 system_prompt_template="", user_prompt_template="", max_word_ratio=0.6):
        super().__init__(model_name, api_version, temperature, max_tokens, system_prompt_template, user_prompt_template)
        self.max_word_ratio = max_word_ratio

    def _get_response(self, messages):
        response = openai.ChatCompletion.create(
            engine=self.model_name,
            api_version=self.api_version,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n
        )

        return response

    def call_llm(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        system_prompt = self.system_prompt_template.format(**{'max_words': max_words})
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text

    def call_llm_with_res(self, input_text):
        input_text_word_count = len(input_text.split())
        max_words = int(round(input_text_word_count * self.max_word_ratio))
        system_prompt = self.system_prompt_template.format(**{'max_words': max_words})
        user_prompt = self.user_prompt_template.format(**{'input_text': input_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._get_response(messages)
        output_text = response["choices"][0]["message"]["content"].strip()
        return output_text, response

    def get_config(self):
        return self.__dict__
